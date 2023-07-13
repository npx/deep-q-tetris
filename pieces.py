import i_piece
import j_piece
import l_piece
import o_piece
import s_piece
import t_piece
import z_piece
from typing import TypeAlias


Piece: TypeAlias = tuple[str, int]

PIECES = {
    "i": i_piece.i_piece,
    "j": j_piece.j_piece,
    "l": l_piece.l_piece,
    "o": o_piece.o_piece,
    "s": s_piece.s_piece,
    "t": t_piece.t_piece,
    "z": z_piece.z_piece,
}

OFFSET_WIDTH = {
    "i": i_piece.i_piece_offset_width,
    "j": j_piece.j_piece_offset_width,
    "l": l_piece.l_piece_offset_width,
    "o": o_piece.o_piece_offset_width,
    "s": s_piece.s_piece_offset_width,
    "t": t_piece.t_piece_offset_width,
    "z": z_piece.z_piece_offset_width,
}

OFFSET_HEIGHT = {
    "i": i_piece.i_piece_offset_height,
    "j": j_piece.j_piece_offset_height,
    "l": l_piece.l_piece_offset_height,
    "o": o_piece.o_piece_offset_height,
    "s": s_piece.s_piece_offset_height,
    "t": t_piece.t_piece_offset_height,
    "z": z_piece.z_piece_offset_height,
}

DISTINCT_ROTATION = {
    "i": i_piece.i_distinct_rotations,
    "j": j_piece.j_distinct_rotations,
    "l": l_piece.l_distinct_rotations,
    "o": o_piece.o_distinct_rotations,
    "s": s_piece.s_distinct_rotations,
    "t": t_piece.t_distinct_rotations,
    "z": z_piece.z_distinct_rotations,
}


def make_piece(name: str) -> Piece:
    return (name, 0)


def get_rotation(piece: Piece, direction: int) -> Piece:
    name, rotation = piece
    return (name, (rotation+direction) % 4)


def get_dim(piece: Piece) -> tuple[int, int]:
    name, rotation = piece
    blocks = get_blocks(piece)
    return (len(blocks[0]), len(blocks))


def get_blocks(piece: Piece) -> list[list[int]]:
    name, rotation = piece
    return PIECES[name][rotation]


def offset_width(piece: Piece) -> tuple[int, int]:
    name, rotation = piece
    return OFFSET_WIDTH[name][rotation]


def offset_height(piece: Piece) -> tuple[int, int]:
    name, rotation = piece
    return OFFSET_HEIGHT[name][rotation]


def get_distinct_rotations(piece: Piece) -> int:
    name, rotation = piece
    return DISTINCT_ROTATION[name]
