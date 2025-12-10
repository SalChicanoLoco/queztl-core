#!/usr/bin/env python3
"""
QuetzalCore VM Console Server
Provides web-based remote access to VM management
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import subprocess
import os
from pathlib import Path

class VMConsoleHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = '/console.html'
        elif self.path == '/api/vm/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Read VM status
            status_file = Path(__file__).parent / 'STATUS'
            status = status_file.read_text().strip() if status_file.exists() else 'unknown'
            
            # Read VM config
            config_file = Path(__file__).parent / 'config.json'
            config = json.loads(config_file.read_text()) if config_file.exists() else {}
            
            response = {
                'status': status,
                'vm_id': config.get('vm_id', 'unknown'),
                'name': config.get('name', 'Unknown VM'),
                'memory_mb': config.get('memory_mb', 0),
                'vcpus': config.get('vcpus', 0),
                'disk_size_gb': config.get('disk_size_gb', 0),
                'network': config.get('network', 'bridge'),
                'features': config.get('features', [])
            }
            
            self.wfile.write(json.dumps(response).encode())
            return
        elif self.path == '/api/vm/start':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Start VM
            result = {'success': True, 'message': 'VM start initiated'}
            try:
                subprocess.run(['bash', 'start.sh'], cwd=Path(__file__).parent, check=True)
            except Exception as e:
                result = {'success': False, 'message': str(e)}
            
            self.wfile.write(json.dumps(result).encode())
            return
        elif self.path == '/api/network/test':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Test network connectivity
            results = {}
            
            try:
                # Test DNS
                dns_result = subprocess.run(['ping', '-c', '2', '8.8.8.8'], 
                                          capture_output=True, text=True, timeout=5)
                results['dns'] = 'connected' if dns_result.returncode == 0 else 'failed'
                
                # Test Ubuntu archives
                archive_result = subprocess.run(['ping', '-c', '2', 'archive.ubuntu.com'], 
                                               capture_output=True, text=True, timeout=5)
                results['ubuntu_archive'] = 'connected' if archive_result.returncode == 0 else 'failed'
            except Exception as e:
                results['error'] = str(e)
            
            self.wfile.write(json.dumps(results).encode())
            return
            
        return SimpleHTTPRequestHandler.do_GET(self)

def run_server(port=9090):
    """Run the VM console web server"""
    vm_dir = Path(__file__).parent
    os.chdir(vm_dir)
    
    server_address = ('', port)
    httpd = HTTPServer(server_address, VMConsoleHandler)
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║       QuetzalCore VM Console Server                        ║
╚════════════════════════════════════════════════════════════╝

🌐 Server running at: http://localhost:{port}
📁 Serving from: {vm_dir}
🖥️  VM: test-vm-001 (QuetzalCore Test VM)

Open your browser and navigate to:
    http://localhost:{port}

Press Ctrl+C to stop the server
    """)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped")
        httpd.server_close()

if __name__ == '__main__':
    run_server()
