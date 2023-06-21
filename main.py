import board
import rotation
import pieces

if __name__ == "__main__":
    tst_board = """
    ..........
    xx..x.....
    x...xxxxxx
    xx.xxxxxxx
    """

    piece = pieces.make_piece("t")
    piece = pieces.get_rotation(piece, -1)
    pos = (0, 2)

    test_board = board.make_board(tst_board)
    board.print_board(test_board)

    placed = board.place_piece(test_board, piece, pos)
    board.print_board(placed)

    direction = -1
    kick = rotation.get_kick(test_board, piece, direction, pos)
    if kick:
        piece = pieces.get_rotation(piece, direction)
        pos = kick

    placed = board.place_piece(test_board, piece, pos)
    board.print_board(placed)

    placed = board.lock(placed)
    board.print_board(placed)

    placed, lines = board.clear_lines(placed)
    board.print_board(placed)

# TODO
# determine if piece can be placed with legitimate moves
