from typing import TypeAlias, Optional
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
from board import print_board

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

    state = FloatTensor(np.append(board, [additional], axis=0))

    return state


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
def step(action: Action, env: Environment, renderer: Optional[bool] = False) -> tuple[float, bool, Environment, Stats]:
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

    board_height = sum(1 if sum(line) > 0 else 0 for line in board)
    cleared_height = sum(1 if sum(line) > 0 else 0 for line in cleared)
    added_lines = cleared_height - board_height

    # TODO scoring
    # TODO end of game
    done = sum(cleared[0]) > 0
    done_bad = -10 if done else 0

    w, h = get_dim(cleared)

    score = 1 + (count ** 3) * w - (added_lines * 2) + done_bad

    stats = (1, count)

    if renderer:
        print_board(placed, True)

    if from_hold:
        new_env = (cleared, pieces[0], new_pieces, next_bag)
    else:
        new_env = (cleared, hold, new_pieces, next_bag)
    return (score, done, new_env, stats)

    #
    # lines_cleared, self.board = self.check_cleared_rows(self.board)
    # score = 1 + (lines_cleared ** 2) * self.width
    # self.score += score
    # self.tetrominoes += 1
    # self.cleared_lines += lines_cleared
    # if not self.gameover:
    #     self.new_piece()
    # if self.gameover:
    #     self.score -= 2
    #
    # return score, self.gameover
    #

# def render(self, video=None):
#     if not self.gameover:
#         img = [self.piece_colors[p] for row in self.get_current_board_state() for p in row]
#     else:
#         img = [self.piece_colors[p] for row in self.board for p in row]
#     img = np.array(img).reshape((self.height, self.width, 3)).astype(np.uint8)
#     img = img[..., ::-1]
#     img = Image.fromarray(img, "RGB")
#
#     img = img.resize((self.width * self.block_size, self.height * self.block_size), 0)
#     img = np.array(img)
#     img[[i * self.block_size for i in range(self.height)], :, :] = 0
#     img[:, [i * self.block_size for i in range(self.width)], :] = 0
#
#     img = np.concatenate((img, self.extra_board), axis=1)
#
#
#     cv2.putText(img, "Score:", (self.width * self.block_size + int(self.block_size / 2), self.block_size),
#                 fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
#     cv2.putText(img, str(self.score),
#                 (self.width * self.block_size + int(self.block_size / 2), 2 * self.block_size),
#                 fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
#
#     cv2.putText(img, "Pieces:", (self.width * self.block_size + int(self.block_size / 2), 4 * self.block_size),
#                 fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
#     cv2.putText(img, str(self.tetrominoes),
#                 (self.width * self.block_size + int(self.block_size / 2), 5 * self.block_size),
#                 fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
#
#     cv2.putText(img, "Lines:", (self.width * self.block_size + int(self.block_size / 2), 7 * self.block_size),
#                 fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
#     cv2.putText(img, str(self.cleared_lines),
#                 (self.width * self.block_size + int(self.block_size / 2), 8 * self.block_size),
#                 fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
#
#     if video:
#         video.write(img)
#
#     cv2.imshow("Deep Q-Learning Tetris", img)
#     cv2.waitKey(1)
