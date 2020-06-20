from tabulate import tabulate
from termcolor import colored


def describe_model(model):
    headers = ['name', 'shape', '#elements', '(M)', '(MB)', 'trainable', 'dtype']

    data = []
    trainable_count = 0
    total = 0
    total_size = 0
    trainable_total = 0

    for name, param in model.named_parameters():
        dtype = str(param.data.dtype)
        param_mb = 'NAN'
        param_bype = float(''.join([s for s in dtype if s.isdigit()])) / 8

        total += param.data.nelement()
        if param_bype:
            param_mb = '{:.02f}'.format(param.data.nelement() / 1024.0 ** 2 * param_bype)
            total_size += param.data.nelement() * param_bype
        data.append([name.replace('.', '/'), list(param.size()),
                     '{0:,}'.format(param.data.nelement()),
                     '{:.02f}'.format(param.data.nelement() / 1024.0 ** 2), param_mb,
                     param.requires_grad, dtype])
        if param.requires_grad:
            trainable_count += 1
            trainable_total += param.data.nelement()

    table = tabulate(data, headers=headers)

    summary_msg = colored(
        "\nNumber of All variables: {}".format(len(data)) +
        "\nAll parameters (elements): {:.02f}M".format(total / 1024.0 ** 2) +
        "\nAll parameters (size): {:.02f}MB".format(total_size / 1024.0 ** 2) +
        "\nNumber of Trainable variables: {}".format(trainable_count) +
        "\nAll trainable parameters (elements): {:.02f}M\n".format(
            trainable_total / 1024.0 ** 2), 'cyan')
    print(colored("List of All Variables: \n", 'cyan') + table + summary_msg)
