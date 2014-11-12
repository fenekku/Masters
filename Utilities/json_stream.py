
"""Deals with json string streams by yielding individual roughly valid json
"""

def json_string_generator(json_stream):
    started = False
    caring = True
    started_index = 0
    json_stream = json_stream.strip()
    idx = 0
    while idx < len(json_stream):
        if not started:
            if caring and json_stream[idx] == "{":
                started = True
                started_index = idx
            idx += 1
        else:
            if caring and json_stream[idx] == "\"":
                caring = False
                idx += 1
            elif not caring and json_stream[idx] == "\\":
                idx += 2 # Skip following character
            elif not caring and json_stream[idx] == "\"":
                caring = True
                idx += 1
            elif caring and json_stream[idx] == "}":
                if idx+1 == len(json_stream):
                    started = False
                    yield json_stream[started_index:]

                elif (json_stream[idx+1] == "\n") or (json_stream[idx+1] == "{"):
                    started = False
                    yield json_stream[started_index:idx+1]

                idx += 1

            else:
                idx += 1