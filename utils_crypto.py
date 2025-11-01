from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os

KEY_LENGTH = 32

def generate_key(password, salt=None):
    """Generates a 32-byte AES-256 key from a user password and salt."""
    if salt is None:
        salt = os.urandom(16)
    password_bytes = password.encode()
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_LENGTH,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = kdf.derive(password_bytes)
    return key, salt

def aes_encrypt(data, key):
    """Encrypts raw bytes using AES-256 in CFB mode, prepending the IV."""
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(data) + encryptor.finalize()
    return iv + ciphertext

def aes_decrypt(encrypted_data, key):
    """Decrypts the data using AES-256."""
    iv = encrypted_data[:16]
    ciphertext = encrypted_data[16:]
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(ciphertext) + decryptor.finalize()
