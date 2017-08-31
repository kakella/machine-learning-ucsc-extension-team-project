import numpy as np
import algorithms.util.constants as cst

def keslerize_column(column_data, negative_value=-1):
    unique_values = sorted(set(column_data))
    num_cols = len(unique_values)

    # print('keslerizing into {} columns'.format(num_cols))

    keslerized_column_data = []

    for v in column_data:
        index = unique_values.index(v)
        keslerized_value = [negative_value] * num_cols
        keslerized_value[index] = +1
        keslerized_column_data.append(keslerized_value)

    return np.array(keslerized_column_data), unique_values


def de_keslerize_columns(keslerized_output_data, unique_values_sorted=None):
    return_value = []

    for d in keslerized_output_data:
        index = d.tolist().index(max(d))
        count_positive_values = sum(x > 0 for x in d)
        if count_positive_values == 0 or count_positive_values > 1:
            return_value.append(cst.INDETERMINATE_VALUE)
        else:
            if unique_values_sorted is None:
                return_value.append(index)
            else:
                return_value.append(unique_values_sorted[index])

    return return_value

