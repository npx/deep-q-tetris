from i_piece import i_piece, i_piece_offset_width, i_piece_offset_height
from j_piece import j_piece, j_piece_offset_width, j_piece_offset_height
from l_piece import l_piece, l_piece_offset_width, l_piece_offset_height
from o_piece import o_piece, o_piece_offset_width, o_piece_offset_height
from s_piece import s_piece, s_piece_offset_width, s_piece_offset_height
from t_piece import t_piece, t_piece_offset_width, t_piece_offset_height
from z_piece import z_piece, z_piece_offset_width, z_piece_offset_height

PIECES = {
    "i": i_piece,
    "j": j_piece,
    "l": l_piece,
    "o": o_piece,
    "s": s_piece,
    "t": t_piece,
    "z": z_piece,
}

OFFSET_WIDTH = {
    "i": i_piece_offset_width,
    "j": j_piece_offset_width,
    "l": l_piece_offset_width,
    "o": o_piece_offset_width,
    "s": s_piece_offset_width,
    "t": t_piece_offset_width,
    "z": z_piece_offset_width,
}

OFFSET_HEIGHT = {
    "i": i_piece_offset_height,
    "j": j_piece_offset_height,
    "l": l_piece_offset_height,
    "o": o_piece_offset_height,
    "s": s_piece_offset_height,
    "t": t_piece_offset_height,
    "z": z_piece_offset_height,
}


def make_piece(name):
    return (name, 0)


def get_rotation(piece, direction):
    name, rotation = piece
    return (name, (rotation+direction) % 4)


def get_dim(piece):
    name, rotation = piece
    blocks = get_blocks(piece)
    return (len(blocks[0]), len(blocks))


def get_blocks(piece):
    name, rotation = piece
    return PIECES[name][rotation]


def offset_width(piece):
    name, rotation = piece
    return OFFSET_WIDTH[name][rotation]


def offset_height(piece):
    name, rotation = piece
    return OFFSET_HEIGHT[name][rotation]
