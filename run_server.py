#!/usr/bin/env python3
"""
HTTP Sunucusu + API - Analiz HTML sayfasını serve etmek ve analiz yapmak için
"""

import http.server
import socketserver
import os
import json
import numpy as np
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from datetime import datetime

PORT = 5000

# Analysis modülünü import et
import sys
sys.path.insert(0, str(Path(__file__).parent))
try:
    from analysis import run_analysis, NaNEncoder
except ImportError:
    from analysis import run_analysis
    NaNEncoder = json.JSONEncoder
import subprocess

# Kayıt klasörü
SAVED_ANALYSES_DIR = Path(__file__).parent / "saved_analyses"
SAVED_ANALYSES_DIR.mkdir(exist_ok=True)

def get_latest_saved_file():
    """En son kaydedilen analiz dosyasını bul"""
    files = list(SAVED_ANALYSES_DIR.glob("analysis_*.json"))
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)

def save_analysis(data, filename=None):
    """Analizi JSON dosyasına kaydet"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{timestamp}.json"
    
    filepath = SAVED_ANALYSES_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=NaNEncoder)
    
    return str(filepath)

def load_analysis(filepath=None):
    """Analizi JSON dosyasından yükle"""
    if filepath is None:
        filepath = get_latest_saved_file()
        if filepath is None:
            return None
    else:
        filepath = Path(filepath)
    
    if not filepath.exists():
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        # NaN ve Infinity değerlerini düzelt
        content = content.replace(': NaN', ': null').replace(':NaN', ':null')
        content = content.replace(': Infinity', ': null').replace(':Infinity', ':null')
        return json.loads(content)

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # CORS headers ekle
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()
    
    def do_OPTIONS(self):
        """OPTIONS isteğine yanıt ver (CORS preflight)"""
        self.send_response(200)
        self.end_headers()
    
    def do_POST(self):
        """POST isteğini işle - Analiz çalıştır"""
        if self.path == '/api/analyze':
            try:
                # İstek boyutunu al
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                
                # JSON verisini parse et
                params = json.loads(body.decode('utf-8'))
                
                start_year = params.get('start_year', 2013)
                end_year = params.get('end_year', 2024)
                happiness_source = params.get('happiness_source', 'ourworldindata')
                economic_indicator = params.get('economic_indicator', 'gdp_pc')
                
                # Parametreleri valide et
                start_year = max(2005, min(2024, int(start_year)))
                end_year = max(start_year, min(2024, int(end_year)))
                
                print(f"📊 Analiz isteği alındı: {start_year}-{end_year} ({happiness_source}, {economic_indicator})")
                
                # Analiz çalıştır
                result = run_analysis(start_year, end_year, happiness_source, economic_indicator)
                
                # Sonucu JSON olarak gönder
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                
                response = json.dumps(result, ensure_ascii=False, indent=2, cls=NaNEncoder)
                response = response.replace('NaN', 'null').replace('Infinity', 'null')
                self.wfile.write(response.encode('utf-8'))
                
            except Exception as e:
                print(f"❌ API Hatası: {e}")
                import traceback
                traceback.print_exc()
                
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                error_response = json.dumps({
                    "success": False,
                    "error": str(e)
                })
                self.wfile.write(error_response.encode('utf-8'))
        
        elif self.path == '/api/country-detail':
            try:
                # İstek boyutunu al
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                
                # JSON verisini parse et
                params = json.loads(body.decode('utf-8'))
                
                iso3 = params.get('iso3', '')
                start_year = params.get('start_year', 2013)
                end_year = params.get('end_year', 2024)
                happiness_source = params.get('happiness_source', 'ourworldindata')
                economic_indicator = params.get('economic_indicator', 'gdp_pc')
                
                if not iso3:
                    raise ValueError("iso3 parametresi gerekli")
                
                print(f"📍 Ülke detay isteği: {iso3} ({start_year}-{end_year}, {economic_indicator})")
                
                # Yıllık verileri al (eski fonksiyon artık mevcut değil)
                result = {
                    "success": False,
                    "error": "Bu API endpoint'i kullanılmıyor"
                }
                
                # Sonucu JSON olarak gönder
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                
                response = json.dumps(result, ensure_ascii=False, indent=2, cls=NaNEncoder)
                response = response.replace('NaN', 'null').replace('Infinity', 'null')
                self.wfile.write(response.encode('utf-8'))
                
            except Exception as e:
                print(f"❌ Country Detail API Hatası: {e}")
                import traceback
                traceback.print_exc()
                
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                error_response = json.dumps({
                    "success": False,
                    "error": str(e)
                })
                self.wfile.write(error_response.encode('utf-8'))
        
        elif self.path == '/api/clustering':
            try:
                # İstek parametrelerini al
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length) if content_length > 0 else b'{}'
                params = json.loads(body.decode('utf-8')) if body else {}
                
                pipeline_type = params.get('pipeline_type', 'both')  # "multivariate", "score_only", "both"
                start_year = params.get('start_year', 2013)
                end_year = params.get('end_year', 2023)
                
                print(f"[*] Kümeleme isteği: {pipeline_type} ({start_year}-{end_year})")
                
                # Dual clustering modülünü import et ve çalıştır
                from dual_clustering import run_dual_clustering
                
                result = run_dual_clustering(start_year, end_year, pipeline_type)
                
                if not result.get('success'):
                    raise RuntimeError(result.get('error', 'Bilinmeyen hata'))
                
                print(f"[+] Kümeleme tamamlandı: {pipeline_type}")
                
                # Başarı yanıtı gönder
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                
                response = json.dumps(result, ensure_ascii=False)
                self.wfile.write(response.encode('utf-8'))
                
            except Exception as e:
                print(f"❌ Kümelendirme API Hatası: {e}")
                import traceback
                traceback.print_exc()
                
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                error_response = json.dumps({
                    "success": False,
                    "error": str(e)
                })
                self.wfile.write(error_response.encode('utf-8'))
        
        elif self.path == '/api/save':
            try:
                # İstek verilerini al
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                data = json.loads(body.decode('utf-8'))
                
                filename = data.get('filename', None)
                
                print(f"💾 Kayıt isteği alındı...")
                
                # Veriyi kaydet
                filepath = save_analysis(data, filename)
                
                print(f"✅ Analiz kaydedildi: {filepath}")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                
                response = json.dumps({
                    "success": True,
                    "filepath": filepath,
                    "message": "Analiz başarıyla kaydedildi"
                }, ensure_ascii=False)
                self.wfile.write(response.encode('utf-8'))
                
            except Exception as e:
                print(f"❌ Kayıt API Hatası: {e}")
                import traceback
                traceback.print_exc()
                
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                error_response = json.dumps({
                    "success": False,
                    "error": str(e)
                })
                self.wfile.write(error_response.encode('utf-8'))
        
        elif self.path == '/api/load':
            try:
                # İstek verilerini al (opsiyonel filepath)
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length) if content_length > 0 else b'{}'
                params = json.loads(body.decode('utf-8')) if body else {}
                
                filepath = params.get('filepath', None)
                
                print(f"📂 Yükleme isteği alındı...")
                
                # Veriyi yükle
                data = load_analysis(filepath)
                
                if data is None:
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json; charset=utf-8')
                    self.end_headers()
                    
                    response = json.dumps({
                        "success": False,
                        "message": "Kaydedilmiş analiz bulunamadı"
                    }, ensure_ascii=False)
                    self.wfile.write(response.encode('utf-8'))
                    return
                
                print(f"✅ Analiz yüklendi")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                
                response = json.dumps({
                    "success": True,
                    "data": data
                }, ensure_ascii=False)
                self.wfile.write(response.encode('utf-8'))
                
            except Exception as e:
                print(f"❌ Yükleme API Hatası: {e}")
                import traceback
                traceback.print_exc()
                
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                error_response = json.dumps({
                    "success": False,
                    "error": str(e)
                })
                self.wfile.write(error_response.encode('utf-8'))
        
        elif self.path == '/api/import':
            try:
                # Dosya içeriğini al
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                data = json.loads(body.decode('utf-8'))
                
                print(f"📥 İçe aktarma isteği alındı...")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                
                response = json.dumps({
                    "success": True,
                    "data": data
                }, ensure_ascii=False)
                self.wfile.write(response.encode('utf-8'))
                
            except Exception as e:
                print(f"❌ İçe Aktarma API Hatası: {e}")
                import traceback
                traceback.print_exc()
                
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                error_response = json.dumps({
                    "success": False,
                    "error": str(e)
                })
                self.wfile.write(error_response.encode('utf-8'))
        
        elif self.path == '/api/list-saved':
            try:
                # Kaydedilmiş dosyaları listele
                files = list(SAVED_ANALYSES_DIR.glob("analysis_*.json"))
                file_list = []
                
                for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
                    file_list.append({
                        "filename": f.name,
                        "filepath": str(f),
                        "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                        "size": f.stat().st_size
                    })
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                
                response = json.dumps({
                    "success": True,
                    "files": file_list
                }, ensure_ascii=False)
                self.wfile.write(response.encode('utf-8'))
                
            except Exception as e:
                print(f"❌ Liste API Hatası: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                error_response = json.dumps({
                    "success": False,
                    "error": str(e)
                })
                self.wfile.write(error_response.encode('utf-8'))
        
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"error": "Not found"}')

# Çalışan dizini değiştir
os.chdir(Path(__file__).parent)

try:
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"""
╔════════════════════════════════════════════════════════════════╗
║    GDELT → IPTC Mapping Dashboard - HTTP Sunucu + API          ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  🌐 Sunucu başlatıldı!                                         ║
║                                                                ║
║  🔗 Sayfayı açmak için: http://localhost:{PORT}                 ║
║  📡 API Endpoint: POST http://localhost:{PORT}/api/analyze      ║
║                                                                ║
║  📁 Dizin: {Path(__file__).parent}                             ║
║                                                                ║
║  ⏹️  Durdurmak için: Ctrl+C                                     ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")
        httpd.serve_forever()
        
except KeyboardInterrupt:
    print("\n\n✋ Sunucu durduruldu.")
except OSError as e:
    print(f"\n❌ Hata: {e}")
    if "Address already in use" in str(e):
        print(f"Port {PORT} zaten kullanımda. Lütfen başka bir program kapatın.")

