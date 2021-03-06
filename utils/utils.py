from pathlib import Path
import torch
import os



def save_dict(params, name, dirname=None):
    if dirname is None:
        dirname = os.path.dirname(os.path.realpath(__file__))
        dirname = dirname.rsplit('/', 1)[0] + '/checkpoints'
        dirname_path = Path(dirname)
        if not dirname_path.is_dir():
            dirname_path.mkdir(parents=True, exist_ok=False)
    else:
        dirname_path = Path(dirname)
        if not dirname_path.is_dir():
            raise NotADirectoryError

    torch.save(params, dirname + '/epoch_' + name)


def load_model_dict_abs(abs_path):
    state_dict = torch.load(abs_path)
    return state_dict


def load_model_dict(name, dirname = None):
    if dirname is None:
        dirname = os.path.dirname(os.path.realpath(__file__))
        dirname = dirname.rsplit('/', 1)[0] + '/checkpoints'
        dirname_path = Path(dirname)
        if not dirname_path.is_dir():
            dirname_path.mkdir(parents=True, exist_ok=False)
    else:
        dirname_path = Path(dirname)
        if not dirname_path.is_dir():
            raise NotADirectoryError

    state_dict = torch.load(dirname + '/' + name)
    return state_dict


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def print_banner():
    print(
        """   
    ───────────────────────┌▄▄██████████▄▄▄▄▄╓──────────────────────
    ────────────────────▄███████████████████████▄╓──────────────────
    ──────────────────╓████████████████▀▀──▀███████─────────────────
    ────────────────┌█████████████████────────▀█████▄───────────────
    ───────────────▄████████████████▀───────────██████──────────────
    ──────────────╒███████████████▀──────────────█████▌─────────────
    ──────────────███████████████▀───────────────└█████─────────────
    ─────────────▐█████████████▀─┌▄▄▄▄┌───────┌╓╓┌└████▄────────────
    ─────────────███████████▀────────└└──────└└└▀▀▀▀████────────────
    ─────────────█████████████▌▌╙╓▄▄▄└─╖╓▄▄┌╒══╗─═══▄██─────────────
    ─────────────████████────▀█────└─└──█▀▀█──└▀└───▐██─────────────
    ─────────────████████▌────└╓─────┌▄█───▀▄──────┌███─────────────
    ─────────────███▀█▌▐█▌────────────────────└└└└└─▐█──────────────
    ─────────────███▄▀──█▌───────────╕──┌───┌───────▀───────────────
    ─────────────▐█████▄▄╕──────────────▀────└─────▐────────────────
    ──────────────▀███████─────────────═─└─└═──────▌────────────────
    ───────────────╙██████▌──────────▄╓╗╥▓╗═─┌──────────────────────
    ────────────────▐█████▌╕──────────└═┌┌┌┌╛─────╛─────────────────
    ─────────────────██████──└╕┌────────────────▄───────────────────
    ────────────────┌═▀▀███─────└─═┌─────────┌▄██───────────────────
    ─────────────┌═└───╒└─│───────────└└└──└──█▀─└═┌────────────────
    ───────┌╒═└──╕─────┐─────────────────────▐──────║┌──────────────
    ───────────────└╕───└╕───────────────────▐────┌┘───└═───────────
    ──────────────────═───└═┌───────────────┌╛──────────────────────
        ─────────────▀▀██▀▀───────────────╓▄──────────▄┌──────────██▌───
    ───────────────────────╒█▄───┌╓──▐██──────────██──┌▄█┐───╒██▌───
    ────────▄▄──────────╓──▐██───╙██▄██▌─▄█▄──▄▄──██─┌██▄███─▐██────
    ───┌▄████▀────▄███──██┐─██─────▀██▀─▐███▌─██▄─██─██─███──██▌────
    ───▀▀▀─██──█▌─████▌▐███▄██──────▐█──█████▄██████─▀█████─▐██─────
    ───────▐██─██─██▀██▐█▌╙██▀───────█▌─█▌─▐██▀▀─▀▀──────▐█═─▀──────
    ─────▄▄▄██─▐█▌─└──▀──────────────██────────────╒─╔───┌╕╥██▄─────
    ─────▀██▀└───────────────────────▀▀────────────────────╙██▀─────
        """
    )

def hotdog():
    print(
        """
        █████████████████████████████████▀█
        █─█─█─▄▄─█─▄─▄─███▄─▄▄▀█─▄▄─█─▄▄▄▄█
        █─▄─█─██─███─██████─██─█─██─█─██▄─█
        ▀▄▀▄▀▄▄▄▄▀▀▄▄▄▀▀▀▀▄▄▄▄▀▀▄▄▄▄▀▄▄▄▄▄▀
        """
    )

def nothotdog():
    print(
        """
        ██████████████████████████████████████████████████████▀█
        █▄─▀█▄─▄█─▄▄─█─▄─▄─███─█─█─▄▄─█─▄─▄─███▄─▄▄▀█─▄▄─█─▄▄▄▄█
        ██─█▄▀─██─██─███─█████─▄─█─██─███─██████─██─█─██─█─██▄─█
        ▀▄▄▄▀▀▄▄▀▄▄▄▄▀▀▄▄▄▀▀▀▀▄▀▄▀▄▄▄▄▀▀▄▄▄▀▀▀▀▄▄▄▄▀▀▄▄▄▄▀▄▄▄▄▄▀
        """
    )