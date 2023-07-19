import cv2
from typing import TypeAlias, Optional, Any
from board import Board, Placement
from board import make_board, place_piece, lock
from pieces import PIECES
import random
from torch import FloatTensor
from board import get_placements
import numpy as np
from pieces import get_distinct_rotations
from pieces import get_rotation
from pieces import make_piece
from board import clear_lines
from board import get_dim
from PIL import Image
from matplotlib import style

style.use("ggplot")


# Board, HoldPiece, CurrentBag, NextBag
# TODO TypedDict
Environment: TypeAlias = tuple[Board, str, list[str], list[str]]
Action: TypeAlias = tuple[Placement, bool] | None
# pieces, lines_cleared
Stats: TypeAlias = tuple[int, int]


def make_bag() -> list[str]:
    names = list(PIECES)
    random.shuffle(names)
    return names


def make_environment(seed: Optional[int] = None) -> Environment:
    if seed is not None:
        random.seed(seed)

    board = make_board()
    hold = ""
    current_bag = make_bag()
    next_bag = make_bag()

    return (board, hold, current_bag, next_bag)


def encode_state(env: Environment) -> FloatTensor:
    """
    after:
    [
            [... board                                      ... ],
            [...   .                                        ... ],
            [...   .                                        ... ],
            [...   .                                        ... ],
            [... board                                      ... ],
            [hold, next1, next2, next3, next4, next5, 0, 0, 0, 0],
    ]
    """

    # TODO This requires min board width of 10... hardcode board size?
    # TODO make next q size and hold configurable
    board, hold, pieces, next_bag = env

    hold_num = -1 if not hold else list(PIECES).index(hold)
    ref = list(PIECES)
    next_queue = [ref.index(name) for name in (pieces + next_bag)[0:5]]
    additional = ([hold_num] + next_queue + [0]*10)[0:10]

    state2D = np.append(board, [additional], axis=0)
    state1D = state2D.flatten()
    return FloatTensor(state1D)


def reset() -> Environment:
    return make_environment()


# TODO how to actually update environment?
def get_next_states(env: Environment) -> dict[Action, FloatTensor]:
    """next_states -> actionmap from action to resulting state"""
    # TODO consider narrowing output states (T piece doesnt go right)
    # TODO DRY
    states: dict[Action, FloatTensor] = {}
    board, hold, pieces, next_bag = env

    # choose next piece
    piece = make_piece(pieces[0])
    rotations = get_distinct_rotations(piece)
    for rot in range(rotations):
        placements = get_placements(board, piece)
        for placement in placements:
            pc, pos = placement
            placed = place_piece(board, piece, pos)
            locked = lock(placed)
            cleared, count = clear_lines(locked)

            env = (board, hold, pieces[1:], next_bag)
            states[(placement, False)] = encode_state(env)
        piece = get_rotation(piece, 1)

    # choose the piece in hold
    if hold:
        piece = make_piece(hold)
        rotations = get_distinct_rotations(piece)
        for rot in range(rotations):
            placements = get_placements(board, piece)
            for placement in placements:
                pc, pos = placement
                placed = place_piece(board, piece, pos)
                locked = lock(placed)
                cleared, count = clear_lines(locked)

                env = (board, pieces[0], pieces[1:], next_bag)
                states[(placement, True)] = encode_state(env)
            piece = get_rotation(piece, 1)

    # hold current piece instead
    else:
        env = (board, pieces[0], pieces[1:], next_bag)
        states[None] = encode_state(env)

    return states


# reward, done
def step(action: Action, env: Environment) -> tuple[float, bool, Environment, Stats]:
    """step => given the "action" return the reward and done"""
    board, hold, pieces, next_bag = env

    new_pieces = pieces[1:]

    if len(new_pieces) < 1:
        new_pieces = next_bag
        next_bag = make_bag()

    if action is None:
        new_env = (board, pieces[0], new_pieces, next_bag)
        stats = (0, 0)
        return (0, False, new_env, stats)

    placement, from_hold = action
    piece, pos = placement
    placed = place_piece(board, piece, pos)
    locked = lock(placed)
    cleared, count = clear_lines(locked)

    done = sum(cleared[0]) + sum(cleared[1]) > 0

    w, h = get_dim(cleared)

    score = 1 + (count ** 2) * 10
    score = score if not done else -2

    stats = (1, count)

    if from_hold:
        new_env = (cleared, pieces[0], new_pieces, next_bag)
    else:
        new_env = (cleared, hold, new_pieces, next_bag)

    return (score, done, new_env, stats)


def render(env: Environment, score: float, stats: Stats, video: Any | None = None) -> None:
    colors = [
        (0, 0, 0),
        (255, 255, 0),
        (147, 88, 254),
        (54, 175, 144),
        (255, 0, 0),
        (102, 217, 238),
        (254, 151, 32),
        (0, 0, 255)
    ]
    text_color = (200, 20, 220)

    board, hold, bag, next_bag = env
    w, h = get_dim(board)
    block_size = 30
    tetrominoes, cleared_lines = stats

    extra_board = np.ones((h * block_size, w * int(block_size / 2), 3),
                          dtype=np.uint8) * np.array([204, 204, 255], dtype=np.uint8)

    img = [colors[p] for row in board for p in row]
    img = np.array(img).reshape((h, w, 3)).astype(np.uint8)
    img = img[..., ::-1]
    img = Image.fromarray(img, "RGB")

    img = img.resize((w * block_size, h * block_size), 0)
    img = np.array(img)
    img[[i * block_size for i in range(h)], :, :] = 0
    img[:, [i * block_size for i in range(w)], :] = 0

    img = np.concatenate((img, extra_board), axis=1)

    cv2.putText(img, "Score:", (w * block_size + int(block_size / 2), block_size),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=text_color)
    cv2.putText(img, str(score),
                (w * block_size +
                 int(block_size / 2), 2 * block_size),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=text_color)

    cv2.putText(img, "Pieces:", (w * block_size + int(block_size / 2), 4 * block_size),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=text_color)
    cv2.putText(img, str(tetrominoes),
                (w * block_size +
                 int(block_size / 2), 5 * block_size),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=text_color)

    cv2.putText(img, "Lines:", (w * block_size + int(block_size / 2), 7 * block_size),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=text_color)
    cv2.putText(img, str(cleared_lines),
                (w * block_size +
                 int(block_size / 2), 8 * block_size),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=text_color)

    if video:
        video.write(img)

    cv2.imshow("Deep Q-Learning Tetris", img)
    cv2.waitKey(1)
