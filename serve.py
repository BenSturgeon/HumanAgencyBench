import os
import time

from flask import Flask, send_from_directory, render_template_string


try:
    from systemd import daemon
    systemd_available = True
except ImportError:
    systemd_available = False
    print("Systemd module not available, watchdog functionality disabled")


app = Flask(__name__)


@app.route('/')
def list_files():
    files = []
    output_dir = 'output/output'
    # Check if 'output' directory exists before walking
    if os.path.isdir(output_dir):
        for root, dirs, filenames in os.walk(output_dir):
            for filename in filenames:
                # Construct the full path
                full_path = os.path.join(root, filename)
                # Get the path relative to the current working directory for linking/serving
                relative_path = os.path.relpath(full_path, os.getcwd())
                if relative_path.endswith('.html') or relative_path.endswith('raw.csv'):
                    files.append(relative_path)

    # Add model comparison PNG to the files list if it exists at the root
    model_comparison_file = 'model_comparison.png'
    if os.path.exists(model_comparison_file):
        files.append(model_comparison_file)

    # Add consistency report HTML to the files list if it exists
    # Assuming it's in 'sensitivity_analysis' relative to cwd based on context
    consistency_report_file = os.path.join('sensitivity_analysis', 'consistency_report.html')
    if os.path.exists(consistency_report_file):
        # Add the relative path for linking/serving
        files.append(os.path.relpath(consistency_report_file, os.getcwd()))
    html = """
    <!DOCTYPE html>
    <html>
    <body>
        <h1>Files in Directory:</h1>
        <ul>
        {% for file in files %}
            <li><a href="{{ file }}">{{ file }}</a></li>
        {% endfor %}
        </ul>
        
        {% if 'model_comparison.png' in files %}
        <h2>Model Comparison:</h2>
        <img src="model_comparison.png" alt="Model Comparison Chart" style="max-width: 100%;">
        {% endif %}
    </body>
    </html>
    """
    return render_template_string(html, files=files)


@app.route('/<path:filename>')
def serve_file(filename):
    if not (filename.endswith(".html") or filename.endswith("raw.csv") or filename.endswith(".png")):
        return "Forbidden", 403
        
    return send_from_directory(os.getcwd(), filename)


@app.route('/health')
def heath():
    return "OK", 200


def watchdog_thread():
    if not systemd_available:
        return
        
    daemon.notify('READY=1')
    
    flask_server_active = True
    
    while True:
        try:
            import requests
            from requests.packages.urllib3.exceptions import InsecureRequestWarning
            
            requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
            
            response = requests.get('https://localhost:5000/health', verify=False, timeout=5)
            
            if response.status_code != 200:
                print(f"Server health check failed: got status {response.status_code}")
                flask_server_active = False
            else:
                flask_server_active = True
                
        except Exception as e:
            print(f"Server health check failed: {str(e)}")
            flask_server_active = False
        
        if flask_server_active:
            daemon.notify('WATCHDOG=1')
        else:
            print("Not sending watchdog ping due to server health check failure")
            
        time.sleep(10)


if __name__ == '__main__':

    if systemd_available:
        import threading
        
        watchdog = threading.Thread(target=watchdog_thread)
        watchdog.start()


    app.run(
        debug=False, 
        host="0.0.0.0", 
        port=5000,
        ssl_context=(
            '/etc/letsencrypt/live/agencyevals.ath.cx/fullchain.pem',
            '/etc/letsencrypt/live/agencyevals.ath.cx/privkey.pem'
        )
    )      