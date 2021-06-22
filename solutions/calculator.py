def get_intersection(bin_img, x1, y1, x2, y2):
    repeatable_x, repeatable_y = x1, y1
    accumulatedValue = x1
    ratio = (x2 - x1) / (y2 -y1)

    while(repeatable_y >= y2):
        if(bin_img[repeatable_y,repeatable_x] == 255):
            return (repeatable_y, repeatable_x)
            repeatable_y -= 1
            accumulatedValue -= ratio
            repeatable_x = int(accumulatedValue)
    return (-1,-1)
