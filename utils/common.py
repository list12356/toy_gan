def get_minibatches(inputs, batch_size):
    """
    input_data:
        An ndarry or list of the data
    output_data:
        A list of minibatches (batch_size * batch_num)
    """
    sample_num = len(inputs)
    minibatch_list = []
    for batch_num in range(self.config.sample_num / batch_size):
        minibatch_list.append(np.reshape(inputs[batch_num * batch_size: (batch_num + 1)* batch_size]))
    return minibatch_list