from copy import deepcopy


def FedAvg(global_net, local_nets):
    n_client = len(local_nets)
    n_layers = len(local_nets[0].state_dict())
    # Initialize the result with the last client's state_dict
    result = local_nets[-1].state_dict()

    # Divide all the weights in the result by the number of clients
    for key in result.keys():
        result[key] = result[key] / n_client

    # Aggregate weights from other clients
    for i in range(n_layers):
        # Using n_client - 1: because result contains already last client's weights
        for j in range(n_client - 1):
            result[list(result.keys())[i]] += local_nets[j].state_dict()[list(result.keys())[i]] / n_client
    """# 将客户端模型参数取平均并更新全局模型
    global_parameter = deepcopy(global_net.state_dict())
    local_parameters = [deepcopy(local_net.state_dict()) for local_net in local_nets]

    for layer_name in global_parameter.keys():
        global_parameter[layer_name] = sum([local_parameter[layer_name] for local_parameter in local_parameters])
        global_parameter[layer_name] = global_parameter[layer_name] / len(local_parameters)"""
    return result
