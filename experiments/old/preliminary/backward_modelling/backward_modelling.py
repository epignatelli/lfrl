import numpy
from numpy import ndarray as Array


GLYPH_OBSCURED = "?"


def upscale_glyphs(glyphs: Array, ratio: int=3) -> Array:
    """
    Upscale the glyphs by a given ratio.

    Parameters
    ----------
    glyphs : Array
        The glyphs to upscale.
    ratio : int
        The ratio to upscale the glyphs by.

    Returns
    -------
    Array
        The upscaled glyphs.
    """
    return numpy.repeat(numpy.repeat(glyphs, ratio, axis=0), ratio, axis=1)


def downscale_glyphs(glyphs: Array, ratio: int=3) -> Array:
    """
    Downscale the glyphs by a given ratio.

    Parameters
    ----------
    glyphs : Array
        The glyphs to downscale.
    ratio : int
        The ratio to downscale the glyphs by.

    Returns
    -------
    Array
        The downscaled glyphs.
    """
    return glyphs[::ratio, ::ratio]


def glyphs_to_array(glyphs: str) -> Array:
    """
    Convert a string of glyphs to an array.

    Parameters
    ----------
    glyphs : str
        The glyphs to convert.

    Returns
    -------
    Array
        The glyphs as an array.
    """
    glyphs = glyphs.replace(" ", GLYPH_OBSCURED)
    glyphs_list = [list(line) for line in glyphs.split("\n") if line]
    max_len = max(len(line) for line in glyphs_list)
    for line in glyphs_list:
        if len(line) < max_len:
            line.extend([GLYPH_OBSCURED] * (max_len - len(line)))

    return numpy.array(glyphs_list)


def replace_obscured(glyphs: str, replacement: str=GLYPH_OBSCURED) -> str:
    """
    Replace obscured glyphs with a given replacement.

    Parameters
    ----------
    glyphs : str
        The glyphs to replace.
    replacement : str
        The replacement for obscured glyphs.

    Returns
    -------
    str
        The glyphs with obscured glyphs replaced.
    """
    return glyphs.replace(" ", replacement)


def array_to_glyphs(array: Array) -> str:
    """
    Convert an array of glyphs to a string.

    Parameters
    ----------
    array : Array
        The array to convert.

    Returns
    -------
    str
        The glyphs as a string.
    """
    lists = array.tolist()
    for i, line in enumerate(lists):
        lists[i] = "".join(line)
    return "\n".join(lists)


def add_separators(glyphs: str, separator: str = " ") -> str:
    """
    Add a separator between each glyph.

    Parameters
    ----------
    glyphs : str
        The glyphs to add a separator to.
    separator : str
        The separator to add.

    Returns
    -------
    str
        The glyphs with a separator added.
    """
    lines = glyphs.split("\n")
    for i, line in enumerate(lines):
        lines[i] = separator.join(line)
        lines[i] += separator
    glyphs = "\n".join(lines)
    return glyphs


def pad_glyphs(glyphs: str) -> str:
    """
    Pad the glyphs with a given amount of padding.

    Parameters
    ----------
    glyphs : str
        The glyphs to pad.
    pad : int
        The amount of padding to add.

    Returns
    -------
    str
        The padded glyphs.
    """
    lines = glyphs.split("\n")
    max_len = max(len(line) for line in lines)
    for i, line in enumerate(lines):
        if len(line) < max_len:
            lines[i] = line + " " * (max_len - len(line))
    glyphs = "\n".join(lines)
    return glyphs


if __name__ == "__main__":
    glyphs = """
............
............
............
............
............
@...........
............
"""

    glyphs = """
.......      .....}
......`   ........
......` .......`
.................
....@.......`
..............`
..................}
"""
    padded = pad_glyphs(glyphs)
    replaced = replace_obscured(padded)
    glyphs = add_separators(replaced)
    print(glyphs)


    # replaced = replace_obscured(glyphs)
    # print("Replaced:\n", replaced)

    # array = glyphs_to_array(glyphs)
    # print("Array shape:\n", array.shape)
    # print("Array:\n", array)

    # upscaled = upscale_glyphs(array, 3)
    # print("Upscaled shape:\n", upscaled.shape)
    # print("Upscaled:\n", upscaled)

    # upscaled_glyphs = array_to_glyphs(upscaled)
    # print("Upscaled glyphs:\n", upscaled_glyphs)



