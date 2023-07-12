import pieces
import os


def make_board(board=None):
    if not board:
        return [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        ]
    return parse_board(board)


def parse_board(board):
    lines = [line.strip() for line in board.strip().splitlines()]
    return list(list(map(lambda x: [1, 0][x == '.'], line)) for line in lines)


def get_dim(board):
    return (len(board[0]), len(board))


def get_slice(board, pos, dim):
    u, v = pos
    w, h = dim
    return [rows[v:v+w] for rows in board[u:u+h]]


def is_piece_in_bounds(board, piece, pos):
    # TODO consider that piece can in fact go out of the top
    wb, hb = get_dim(board)
    wp, hp = pieces.get_dim(piece)
    row, col = pos
    blocks = pieces.get_blocks(piece)

    over_left = min(abs(min(0, col)), wp)
    left_oob = get_slice(blocks, (0, 0), (over_left, hp))

    over_top = min(abs(min(row, 0)), hp)
    top_oob = get_slice(blocks, (0, 0), (wp, over_top))

    over_right = min(max(col+wp-wb, 0), wp)
    right_oob = get_slice(blocks, (0, wp-over_right), (over_right, hp))

    over_bottom = min(max(row+hp-hb, 0), hp)
    bot_oob = get_slice(blocks, (hp-over_bottom, 0), (wp, over_bottom))

    in_bounds = sum(sum(v) for v in left_oob + top_oob + right_oob + bot_oob)

    return not in_bounds


def can_place(board, piece, pos):
    if not is_piece_in_bounds(board, piece, pos):
        return False

    w, h = get_dim(board)
    blocks = pieces.get_blocks(piece)

    u, v = pos
    for r, row in enumerate(blocks):
        for c, value in enumerate(row):
            ur = u+r
            vc = v+c
            if 0 <= ur < h and 0 <= vc < w and value > 0 and board[ur][vc] > 0:
                return False

    return True


def print_board(board, clear=False):
    w, h = get_dim(board)
    if clear:
        os.system('clear')
    print("  ╭─" + "─" * w + "─╮")
    for r, row in enumerate(board):
        n = f'{r:02}'
        print(n + "│ " + "".join([' ', '󰿦', '', '%'][v] for v in row) + " │")
    print("  ╰─" + "─" * w + "─╯")


def place_piece(board, piece, pos):
    if not can_place(board, piece, pos):
        raise Exception("cannot place piece")

    blocks = pieces.get_blocks(piece)
    u, v = pos
    board_copy = [x[:] for x in board]
    w, h = get_dim(board)

    for r, row in enumerate(blocks):
        for c, col in enumerate(row):
            tu, tv = u + r, v + c
            if 0 <= tu < h and 0 <= tv < w:
                board_copy[tu][tv] += 2 * col

    return board_copy


def clear_lines(board):
    w, h = get_dim(board)

    cleared_board = list(filter(lambda row: row.count(0) > 0, board))
    cleared_lines = h - len(cleared_board)

    fill = [[0] * w] * cleared_lines

    new_board = fill + cleared_board

    return (new_board, cleared_lines)


def lock(board):
    board_copy = [x[:] for x in board]

    for r, row in enumerate(board_copy):
        for c, col in enumerate(row):
            board_copy[r][c] = not not board_copy[r][c] and True

    return board_copy


def get_placements(board, piece):
    w, h = get_dim(board)
    wp, hp = pieces.get_dim(piece)
    owl, owr = pieces.offset_width(piece)
    oht, ohb = pieces.offset_height(piece)

    actual_piece_width = wp - owl - owr

    col_from = -owl
    col_to = col_from + w - actual_piece_width

    placements = []

    def check(pos): return can_place(board, piece, pos)

    for c in range(col_from, col_to + 1):
        for r in range(-oht, h):
            pos = (r, c)
            next = (r + 1, c)
            if check(pos) and not check(next):
                placements.append((piece, pos))

    return placements


if __name__ == "__main__":
    board = make_board()
    piece = pieces.make_piece("o")

    for _ in range(pieces.get_distinct_rotations(piece)):
        for p in get_placements(board, piece):
            _, pos = p
            placed = place_piece(board, piece, pos)
            print_board(placed, True)
            print(p)
            input()
        piece = pieces.get_rotation(piece, 1)
