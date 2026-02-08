import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")

import jieba
import logging
jieba.setLogLevel(logging.ERROR)

import socket
import os
import sys
from corrector import SentenceCorrector

def start_server(socket_path="/tmp/corrector.sock"):
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    m_path = os.path.join(base_dir, "ngram.pkl")
    l_path = os.path.join(base_dir, "token_dict.json")

    corrector = SentenceCorrector(m_path, l_path)

    if os.path.exists(socket_path):
        os.remove(socket_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(5)
    os.chmod(socket_path, 0o666)

    while True:
        conn, _ = server.accept()
        try:
            data = conn.recv(2048).decode('utf-8')
            if data:
                result, _, _ = corrector.correct(data)
                conn.sendall(result.encode('utf-8'))
        except Exception:
            pass
        finally:
            conn.close()

if __name__ == '__main__':
    start_server()