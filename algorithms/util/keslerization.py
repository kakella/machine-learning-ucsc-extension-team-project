import numpy as np


def keslerize_column(column_data):
    unique_values = sorted(set(column_data))
    num_cols = len(unique_values)

    # print('keslerizing into {} columns'.format(num_cols))

    keslerized_column_data = []

    for v in column_data:
        index = unique_values.index(v)
        keslerized_value = [-1] * num_cols
        keslerized_value[index] = +1
        keslerized_column_data.append(keslerized_value)

    return np.array(keslerized_column_data), unique_values


def de_keslerize_columns(keslerized_output_data, unique_values_sorted=None):
    indexed_list = [d.tolist().index(max(d)) for d in keslerized_output_data]
    if unique_values_sorted is None:
        return np.array(indexed_list)
    else:
        return np.array([unique_values_sorted[d] for d in indexed_list])

