import pieces


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
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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

    over_bottom = min(max(row+hp-wb, 0), hp)
    bot_oob = get_slice(blocks, (hp-over_bottom, 0), (wp, over_bottom))

    in_bounds = sum(sum(v) for v in left_oob + top_oob + right_oob + bot_oob)

    return not in_bounds


def can_place(board, piece, pos):
    if not is_piece_in_bounds(board, piece, pos):
        return False

    dim = pieces.get_dim(piece)
    blocks = pieces.get_blocks(piece)
    slice = get_slice(board, pos, dim)

    for brow, prow in zip(slice, blocks):
        for bval, pval in zip(brow, prow):
            if pval > 0 and bval > 0:
                return False

    return True


def print_board(board):
    for row in board:
        print("".join([' ', '󰿦', '', '%'][v] for v in row))
    print("")


def place_piece(board, piece, pos):
    blocks = pieces.get_blocks(piece)
    u, v = pos
    board_copy = [x[:] for x in board]

    for r, row in enumerate(blocks):
        for c, col in enumerate(row):
            board_copy[u + r][v + c] += 2 * col

    return board_copy
