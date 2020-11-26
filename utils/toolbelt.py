import IPython


def is_running_on_colab():
    """ Returns True if running on Colab"""
    if 'google.colab' in str(IPython.get_ipython()):
        print('Running on CoLab')
        return True
    else:
        print('Not running on CoLab')
        return False


def is_running_locally():
    return is_running_on_colab() is False
