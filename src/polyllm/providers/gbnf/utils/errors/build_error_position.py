MAXIMUM_NUMBER_OF_ERROR_LINES_TO_SHOW = 3


def build_error_position(src: str, pos: int) -> list[str]:
    if src == "":
        return [
            "No input provided",
        ]
    lines = src.split("\n")

    line_idx = 0
    while lines[line_idx] and pos > len(lines[line_idx]) - 1 and pos < len(src):
        pos -= len(lines[line_idx])
        line_idx += 1

    lines_to_show = [
        lines[i]
        for i in range(
            max(0, line_idx - (MAXIMUM_NUMBER_OF_ERROR_LINES_TO_SHOW - 1)),
            line_idx + 1,
        )
    ]

    return [
        *lines_to_show,
        " " * pos + "^",
    ]

def build_error_position(src: str, pos: int) -> list[str]:
    if src == "":
        return [
            "No input provided",
        ]

    if pos >= len(src):
        lines = src.splitlines()
        if lines and lines[-1]:
            return [lines[-1], " " * len(lines[-1]) + "^"]
        elif lines:
            return [lines[-1], "^"]
        else:
            return ["^"]

    lines = src.splitlines(keepends=True) # Keep the newline characters
    line_idx = 0
    current_pos = 0
    error_line = None

    for i, line in enumerate(lines):
        if current_pos <= pos < current_pos + len(line):
            line_idx = i
            error_line = line
            break
        current_pos += len(line)

    if error_line is None:
        # This should ideally not happen if the initial check is correct
        return ["Error position outside of source"]

    # ... (rest of the function to extract surrounding lines and add the '^')
    start_line = max(0, line_idx - (MAXIMUM_NUMBER_OF_ERROR_LINES_TO_SHOW - 1))
    end_line = line_idx + 1
    lines_to_show = [lines[i].rstrip('\n') for i in range(start_line, end_line) if i < len(lines)]

    offset = pos - (sum(len(l) for l in lines[:line_idx]))

    return [
        *lines_to_show,
        " " * offset + "^",
    ]
