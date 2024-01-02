def is_sackgasse(agent_pos, field): #extrem hässlig hardgecoded
    x = agent_pos[0]
    y = agent_pos[1]
    right = 0
    left = 0
    up = 0
    down = 0



    if field[x+1,y]==1 or field[x+1,y]==-1:
        down=0
    else:
        if field[x+1,y+1] == 0 or field[x+1,y-1] == 0:
            down = 0
        else:
            if field[x+2,y]==1 or field[x+2,y]==-1:
                down = 1
            else:
                if field[x+2,y+1] == 0 or field[x+2,y-1] == 0:
                    down = 0
                else:
                    if field[x +3, y] == 1 or field[x + 3, y] == -1:
                        down = 1
                    else:
                        if field[x + 3, y + 1] == 0 or field[x + 3, y - 1] == 0:
                            down = 0
                        else:
                            if field[x +4, y] == 1 or field[x + 4, y] == -1:
                                down = 1
                            else:
                                if field[x + 4, y + 1] == 0 or field[x + 4, y - 1] == 0:
                                    down = 0
                                else:
                                    if field[x + 5, y] == 1 or field[x + 5, y] == -1:
                                        down = 1


    if field[x-1,y]==1 or field[x-1,y]==-1:
        up=0
    else:
        if field[x-1,y+1] == 0 or field[x-1,y-1] == 0:
            up = 0
        else:
            if field[x-2,y]==1 or field[x-2,y]==-1:
                up = 1
            else:
                if field[x-2,y+1] == 0 or field[x-2,y-1] == 0:
                    up = 0
                else:
                    if field[x -3, y] == 1 or field[x - 3, y] == -1:
                        up = 1
                    else:
                        if field[x - 3, y + 1] == 0 or field[x - 3, y - 1] == 0:
                            up = 0
                        else:
                            if field[x -4, y] == 1 or field[x - 4, y] == -1:
                                up = 1
                            else:
                                if field[x - 4, y + 1] == 0 or field[x - 4, y - 1] == 0:
                                    up = 0
                                else:
                                    if field[x - 5, y] == 1 or field[x - 5, y] == -1:
                                        up = 1


    if field[x,y+1]==1 or field[x,y+1]==-1:
        right=0
    else:
        if field[x+1,y+1] == 0 or field[x-1,y+1] == 0:
            right = 0
        else:
            if field[x,y+2]==1 or field[x,y+2]==-1:
                right = 1
            else:
                if field[x+1,y+2] == 0 or field[x-1,y+2] == 0:
                    right = 0
                else:
                    if field[x, y+3] == 1 or field[x, y+3] == -1:
                        right = 1
                    else:
                        if field[x+1, y+3] == 0 or field[x-1, y+3] == 0:
                            right = 0
                        else:
                            if field[x, y+4] == 1 or field[x, y+4] == -1:
                                right = 1
                            else:
                                if field[x + 1, y + 4] == 0 or field[x - 1, y + 4] == 0:
                                    right = 0
                                else:
                                    if field[x, y + 5] == 1 or field[x, y + 5] == -1:
                                        right = 1


    if field[x,y-1]==1 or field[x,y-1]==-1:
        left=0
    else:
        if field[x+1,y-1] == 0 or field[x-1,y-1] == 0:
            left = 0
        else:
            if field[x,y-2]==1 or field[x,y-2]==-1:
                left = 1
            else:
                if field[x+1,y-2] == 0 or field[x-1,y-2] == 0:
                    left = 0
                else:
                    if field[x, y-3] == 1 or field[x, y-3] == -1:
                        left = 1
                    else:
                        if field[x+1, y-3] == 0 or field[x-1, y-3] == 0:
                            left = 0
                        else:
                            if field[x, y-4] == 1 or field[x, y-4] == -1:
                                left = 1
                            else:
                                if field[x + 1, y - 4] == 0 or field[x - 1, y - 4] == 0:
                                    left = 0
                                else:
                                    if field[x, y - 5] == 1 or field[x, y - 5] == -1:
                                        left = 1


    return (up, right, down, left)
