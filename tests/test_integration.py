import subprocess
import time
import sys
import os
import requests
import pytest


def test_integration_server_responds():
    """Start the Flask app in a subprocess and call the /api/sentiment endpoint."""
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    # Start server without Flask reloader (debug=False)
    proc = subprocess.Popen([
        sys.executable, '-u', '-c', 'from app import app; app.run(debug=False, host="127.0.0.1", port=5000)'
    ], env=env)

    try:
        # Wait for server to start
        for _ in range(30):
            try:
                r = requests.post('http://127.0.0.1:5000/api/sentiment', json={'text': 'I love this product!'})
                if r.status_code == 200:
                    data = r.json()
                    assert 'sentiment' in data and 'score' in data
                    return
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        pytest.fail('Server did not respond within timeout')
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
