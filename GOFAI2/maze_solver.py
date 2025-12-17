from collections import deque

FREE = 0
WALL = 1

def in_bounds(maze, r, c):
    return 0 <= r < len(maze) and 0 <= c < len(maze[0])

def is_free(maze, r, c):
    return maze[r][c] == FREE

def neighbors(maze, pos):
    r, c = pos
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)] #UP, DOWN, LEFT, RIGHT
    result = []
    for dr, dc in moves:
        nr, nc = r + dr, c + dc
        if in_bounds(maze, nr, nc) and is_free(maze, nr, nc):
            result.append((nr, nc))

    return result

def reconstruct_path(parent, start, goal):
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]

    path.reverse()
    if path and path[0] == start:
        return path
    return []

def bfs_shortest_path(maze, start, exits):
    queue = deque([start])
    visited = set([start])
    parent = {start: None}

    if start in exits:
        return [start]

    while queue:
        current = queue.popleft()

        for nxt in neighbors(maze, current):
            if nxt in visited:
                continue
            visited.add(nxt)
            parent[nxt] = current

            if nxt in exits:
                return reconstruct_path(parent, start, nxt)

            queue.append(nxt)

    return []

def print_maze_with_path(maze, path, start, exits):
    """
    Stampa il labirinto con:
      # = muro
      . = libero
      S = start
      E = exit
      * = percorso
    """

    path_set = set(path)
    for r in range(len(maze)):
        row_chars = []
        for c in range(len(maze[0])):
            pos = (r, c)
            if pos == start:
                row_chars.append("S")
            elif pos in exits:
                row_chars.append("E")
            elif maze[r][c] == WALL:
                row_chars.append("#")
            elif pos in path_set:
                row_chars.append("*")
            else:
                row_chars.append(".")

        print(" ".join(row_chars))

if __name__ == '__main__':
    maze = [
        [0,0,1,0,0],
        [1,0,1,0,1],
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,0,0,1,0],
    ]
    start = (0, 0)
    exits = {(0,4),(4,1)}

    path = bfs_shortest_path(maze, start, exits)

    if path:
        print("Percorso trovato, lunghezza: ", len(path)-1)
        print("Path: ", path)
    else:
        print("Percorso non trovato")

    print_maze_with_path(maze, path, start, exits)
