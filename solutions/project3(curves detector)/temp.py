import copy


def get_min_max_of_x_or_y(all_point_list, is_horizontal):
    if is_horizontal:
        index = 0  # index of x
    else:
        index = 1  # index of y

    max_val = max(all_point_list, key=lambda point: point[index])[index]
    min_val = min(all_point_list, key=lambda point: point[index])[index]

    return min_val, max_val


def separate(filtered, max_distance, index):
    copied = copy.deepcopy(filtered)
    separated = []
    temp = []
    # TODO 변수 index, idx, filtered, max_distance 이름 다시 짓기
    # 불명확하고, max_distance는 main의 동명의 변수와 겹침

    idx = abs(index - 1)

    for i in copied:
        if len(temp) == 0:
            temp.append(i)
        elif abs(temp[-1][idx] - i[idx]) <= max_distance:
            temp.append(i)
        else:
            separated.append(temp)
            temp = [i]
    if len(temp) > 0:
        separated.append(temp)
    return separated


def flatten_on_one_x_or_y(filtered, index):
    idx = abs(index - 1)
    sum_val = 0
    for i in filtered:
        sum_val += i[idx]
    avg = round(sum_val / len(filtered))
    result = [0, 0]
    result[index] = filtered[0][index]
    result[idx] = avg

    return [result]


# TODO max_distance를  실행 시 두 줄이 한 줄로 나타나는
def flatten(all_point_list, max_distance, min_val, max_val, is_horizontal):
    flattened = []

    for i in range(min_val, max_val + 1):
        if is_horizontal:
            index = 0  # index of x
        else:
            index = 1  # index of y

        flattened_on_one_x_or_y = list(filter(lambda point: point[index] == i, all_point_list))

        if len(flattened_on_one_x_or_y) == 0:
            continue
        elif len(flattened_on_one_x_or_y) == 1:
            flattened = flattened + flattened_on_one_x_or_y
        else:
            a = [[176, 210], [176, 222], [176, 223], [176, 224]]
            b = separate(flattened_on_one_x_or_y, 0, 0)
            c = separate(flattened_on_one_x_or_y, 1, 0)
            d = separate(flattened_on_one_x_or_y, 11, 0)
            e = separate(flattened_on_one_x_or_y, 12, 0)
            f = separate(flattened_on_one_x_or_y, 13, 0)

            separated = separate(flattened_on_one_x_or_y, max_distance, index)
            # if len(separated) > 1:
            for i in separated:
                flattened += flatten_on_one_x_or_y(i, index)


a = [[176, 210], [176, 222], [176, 223], [176, 224]]
b = separate(a, 0, 0)
c = separate(a, 1, 0)
d = separate(a, 11, 0)
e = separate(a, 12, 0)
f = separate(a, 13, 0)
print("b", b)
print("c", c)
print("d", d)
print("e", e)
print("f", f)
