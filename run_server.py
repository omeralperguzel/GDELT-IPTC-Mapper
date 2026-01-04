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

PORT = 8005

# Results directory
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Analysis modülünü import et
import sys
sys.path.insert(0, str(Path(__file__).parent))
try:
    from analysis import run_analysis, NaNEncoder
except ImportError:
    from analysis import run_analysis
    NaNEncoder = json.JSONEncoder
import subprocess

# IPTC treemap modülünü import et
try:
    from export_treemap_tikz import export_treemap_tikz, generate_iptc_treemaps
except ImportError:
    print("  export_treemap_tikz modülü bulunamadı - IPTC treemap API'si çalışmayacak")
    export_treemap_tikz = None
    generate_iptc_treemaps = None

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
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        filename = f"gdelt_analysis_{timestamp}.json"

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

    def do_GET(self):
        # Provide a few API endpoints via GET for convenience (frontend may call via GET)
        if self.path == '/api/list-saved-analyses':
            try:
                files = list(SAVED_ANALYSES_DIR.glob("gdelt_analysis_*.json"))
                file_list = [f.name for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)]

                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()

                response = json.dumps(file_list, ensure_ascii=False)
                self.wfile.write(response.encode('utf-8'))
                return
            except Exception as e:
                print(f" List Saved Analyses (GET) Hatası: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))
                return

        # Fallback to default file serving
        # Wrap the base handler in try/except to avoid full tracebacks when
        # the client requests special files (e.g. .well-known entries) that are
        # absent or when the client aborts the connection while the server is
        # preparing the error response.
        try:
            return super().do_GET()
        except FileNotFoundError:
            try:
                self.send_response(404)
                # respond with a small JSON body to make it easy for the frontend
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(b'{"error": "File not found"}')
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                # client disconnected; nothing to do
                pass
            except Exception:
                # swallow any other write errors
                pass
            return
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            # Client aborted the connection while the base handler was working.
            # Nothing to do here; avoid printing a stack trace.
            return
        except Exception as e:
            # For any other unexpected exception, log and return a 500 JSON
            print(f"Unhandled error serving GET {self.path}: {e}")
            try:
                self.send_response(500)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))
            except Exception:
                pass
            return
    
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
                
                print(f" Analiz isteği alındı: {start_year}-{end_year} ({happiness_source}, {economic_indicator})")
                
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
                print(f" API Hatası: {e}")
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
                
                print(f" Ülke detay isteği: {iso3} ({start_year}-{end_year}, {economic_indicator})")
                
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
                print(f" Country Detail API Hatası: {e}")
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
                print(f" Kümelendirme API Hatası: {e}")
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
                
                print(f" Kayıt isteği alındı...")
                
                # Veriyi kaydet
                filepath = save_analysis(data, filename)
                
                print(f" Analiz kaydedildi: {filepath}")
                
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
                print(f" Kayıt API Hatası: {e}")
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
                
                print(f" Yükleme isteği alındı...")
                
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
                
                print(f" Analiz yüklendi")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                
                response = json.dumps({
                    "success": True,
                    "data": data
                }, ensure_ascii=False)
                self.wfile.write(response.encode('utf-8'))
                
            except Exception as e:
                print(f" Yükleme API Hatası: {e}")
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
                
                print(f" İçe aktarma isteği alındı...")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                
                response = json.dumps({
                    "success": True,
                    "data": data
                }, ensure_ascii=False)
                self.wfile.write(response.encode('utf-8'))
                
            except Exception as e:
                print(f" İçe Aktarma API Hatası: {e}")
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
        
        elif self.path == '/api/run-iptc-mapping':
            try:
                # Istek parametrelerini al
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length) if content_length > 0 else b'{}'
                params = json.loads(body.decode('utf-8')) if body else {}
                
                algorithm = params.get('algorithm', 'v3')  # Default: v3 (hierarchical subtopics)
                mapping_source = params.get('mapping_source', 'vargo')  # Default: vargo
                
                # Scripti sec
                if algorithm == 'v1':
                    script_name = 'gdelt_iptc_mapping.py'
                    json_filename = f'gdelt_iptc_mapping_v1_{mapping_source}.json'
                    print(f"[*] V1 Algoritma: Sadece Embedding ({mapping_source.upper()} source)")
                elif algorithm == 'v2':
                    script_name = 'gdelt_iptc_mapping_v2.py'
                    json_filename = f'gdelt_iptc_mapping_v2_{mapping_source}.json'
                    print(f"[*] V2 Algoritma: Iki Katmanli Fusion ({mapping_source.upper()} source)")
                elif algorithm == 'v3':
                    script_name = 'gdelt_iptc_mapping_v3.py'
                    json_filename = f'gdelt_iptc_mapping_v3_{mapping_source}.json'
                    print(f"[*] V3 Algoritma: Hierarchical Subtopics ({mapping_source.upper()} source)")
                else:
                    raise ValueError(f"Bilinmeyen algoritma: {algorithm}")
                
                script_path = Path(__file__).parent / script_name
                
                if not script_path.exists():
                    raise FileNotFoundError(f"{script_name} bulunamadi")
                
                print(f"[1/5] {script_name} calistiriliyor ({mapping_source} source)...")
                
                result = subprocess.run(
                    ['python', str(script_path), '--mapping-source', mapping_source],
                    capture_output=True,
                    text=True,
                    cwd=str(Path(__file__).parent),
                    encoding='utf-8',
                    errors='replace'
                )
                
                # Terminal ciktisini goster
                if result.stdout:
                    for line in result.stdout.split('\n'):
                        if line.strip():
                            print(f"    {line}")
                
                if result.returncode != 0:
                    print(f"[X] Script hatasi: {result.stderr}")
                    raise RuntimeError(result.stderr)
                
                print(f"[OK] {algorithm.upper()} Mapping tamamlandi ({mapping_source} source)")
                
                # JSON sonuclarini yukle (versiyon-spesifik dosya) under results/
                json_path_root = Path(__file__).parent / json_filename
                json_path_result = RESULTS_DIR / json_filename
                mapping_data = None
                if json_path_result.exists():
                    with open(json_path_result, 'r', encoding='utf-8') as f:
                        mapping_data = json.load(f)
                elif json_path_root.exists():
                    # older scripts may have written to repo root; move into results/ folder
                    with open(json_path_root, 'r', encoding='utf-8') as f:
                        mapping_data = json.load(f)
                    try:
                        json_path_root.replace(json_path_result)
                    except Exception:
                        # fallback: write a copy
                        with open(json_path_result, 'w', encoding='utf-8') as fw:
                            json.dump(mapping_data, fw, ensure_ascii=False, indent=2)
                else:
                    mapping_data = {"success": True, "message": "Mapping tamamlandi", "algorithm": algorithm, "mapping_source": mapping_source}

                # Ensure metadata exists and annotate with algorithm/source and generation time
                if not isinstance(mapping_data, dict):
                    mapping_data = {"data": mapping_data}
                mapping_data.setdefault('metadata', {})
                mapping_data['metadata']['algorithm'] = algorithm
                mapping_data['metadata']['algorithm_name'] = {
                    'v1': 'Embedding Only',
                    'v2': 'Two-Layer Fusion',
                    'v3': 'Hierarchical Subtopics'
                }.get(algorithm, 'Unknown')
                mapping_data['metadata']['mapping_source'] = mapping_source
                mapping_data['metadata'].setdefault('generated_at', datetime.utcnow().isoformat() + 'Z')
                # Persist canonical copy under results folder
                try:
                    with open(json_path_result, 'w', encoding='utf-8') as f:
                        json.dump(mapping_data, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                
                response = json.dumps(mapping_data, ensure_ascii=False, indent=2)
                self.wfile.write(response.encode('utf-8'))
                
            except Exception as e:
                print(f"[X] IPTC Mapping API Hatasi: {e}")
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
        
        elif self.path == '/api/save-analysis':
            try:
                # İstek verilerini al
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                data = json.loads(body.decode('utf-8'))

                filename = data.get('filename', None)
                analysis_data = data.get('data', {})

                print(f" Analiz kaydetme isteği alındı: {filename}")

                # Veriyi kaydet
                filepath = save_analysis(analysis_data, filename)

                print(f" Analiz kaydedildi: {filepath}")

                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()

                response = json.dumps({
                    "success": True,
                    "filepath": str(filepath),
                    "message": "Analiz başarıyla kaydedildi"
                }, ensure_ascii=False)
                self.wfile.write(response.encode('utf-8'))

            except Exception as e:
                print(f" Save Analysis API Hatası: {e}")
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

        elif self.path.startswith('/api/load-analysis/'):
            try:
                # Dosya adını URL'den çıkar
                filename = self.path.replace('/api/load-analysis/', '')

                print(f" Analiz yükleme isteği: {filename}")

                # Dosya yolunu oluştur
                filepath = SAVED_ANALYSES_DIR / filename

                # Veriyi yükle
                data = load_analysis(filepath)

                if data is None:
                    self.send_response(404)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()

                    response = json.dumps({
                        "success": False,
                        "message": "Dosya bulunamadı"
                    }, ensure_ascii=False)
                    self.wfile.write(response.encode('utf-8'))
                    return

                print(f" Analiz yüklendi: {filename}")

                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()

                response = json.dumps(data, ensure_ascii=False, indent=2)
                self.wfile.write(response.encode('utf-8'))

            except Exception as e:
                print(f" Load Analysis API Hatası: {e}")
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

        elif self.path == '/api/list-saved-analyses':
            try:
                # Kaydedilmiş dosyaları listele
                files = list(SAVED_ANALYSES_DIR.glob("gdelt_analysis_*.json"))
                file_list = []

                for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
                    file_list.append(f.name)

                print(f" {len(file_list)} kaydedilmiş analiz bulundu")

                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()

                response = json.dumps(file_list, ensure_ascii=False)
                self.wfile.write(response.encode('utf-8'))

            except Exception as e:
                print(f" List Saved Analyses API Hatası: {e}")
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

        elif self.path.startswith('/api/delete-analysis/'):
            try:
                # Dosya adını URL'den çıkar
                filename = self.path.replace('/api/delete-analysis/', '')

                print(f" Analiz silme isteği: {filename}")

                # Dosya yolunu oluştur
                filepath = SAVED_ANALYSES_DIR / filename

                if filepath.exists():
                    filepath.unlink()
                    print(f" Analiz silindi: {filename}")

                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()

                    response = json.dumps({
                        "success": True,
                        "message": "Analiz başarıyla silindi"
                    }, ensure_ascii=False)
                    self.wfile.write(response.encode('utf-8'))
                else:
                    self.send_response(404)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()

                    response = json.dumps({
                        "success": False,
                        "message": "Dosya bulunamadı"
                    }, ensure_ascii=False)
                    self.wfile.write(response.encode('utf-8'))

            except Exception as e:
                print(f" Delete Analysis API Hatası: {e}")
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
        
        elif self.path == '/api/run-clustering':
            try:
                # İstek parametrelerini al
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length) if content_length > 0 else b'{}'
                params = json.loads(body.decode('utf-8')) if body else {}
                
                mapping_source = params.get('mapping_source', 'vargo')  # 'vargo' veya 'gkg'
                
                print(f"[*] Kümeleme isteği: {mapping_source} kaynaklı")
                
                # Script yolunu kontrol et
                script_path = Path(__file__).parent / 'gdelt_theme_clustering.py'
                if not script_path.exists():
                    raise FileNotFoundError("gdelt_theme_clustering.py bulunamadı")
                
                print(f"[1/3] Kümeleme scripti çalıştırılıyor ({mapping_source})...")
                
                # Scripti çalıştır
                result = subprocess.run(
                    ['python', str(script_path)],
                    capture_output=True,
                    text=True,
                    cwd=str(Path(__file__).parent),
                    encoding='utf-8',
                    errors='replace'
                )
                
                # Terminal çıktısını göster
                if result.stdout:
                    for line in result.stdout.split('\n'):
                        if line.strip():
                            print(f"    {line}")
                
                if result.returncode != 0:
                    print(f"[X] Script hatası: {result.stderr}")
                    raise RuntimeError(result.stderr)
                
                print(f"[OK] Kümeleme tamamlandı: {mapping_source}")
                
                # JSON sonuçlarını yükle
                json_filename = f'gdelt_theme_clusters_{mapping_source}.json'
                json_path = Path(__file__).parent / json_filename
                
                if json_path.exists():
                    with open(json_path, 'r', encoding='utf-8') as f:
                        clustering_data = json.load(f)
                    
                    # Küme sayısını hesapla
                    cluster_count = len(clustering_data.get('cluster_labels', {}))
                    
                    response_data = {
                        "success": True,
                        "message": f"Kümeleme tamamlandı ({mapping_source})",
                        "mapping_source": mapping_source,
                        "cluster_count": cluster_count,
                        "cluster_labels": clustering_data.get('cluster_labels', {}),
                        "themes": clustering_data.get('themes', []),
                        "metadata": clustering_data.get('metadata', {})
                    }
                else:
                    response_data = {
                        "success": True,
                        "message": f"Kümeleme tamamlandı ({mapping_source})",
                        "mapping_source": mapping_source,
                        "cluster_count": 0
                    }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                
                response = json.dumps(response_data, ensure_ascii=False, indent=2)
                self.wfile.write(response.encode('utf-8'))
                
            except Exception as e:
                print(f"[X] Kümeleme API Hatası: {e}")
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
        
        elif self.path == '/api/generate-iptc-treemap':
            try:
                # İstek parametrelerini al
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length) if content_length > 0 else b'{}'
                params = json.loads(body.decode('utf-8')) if body else {}
                
                iptc_category = params.get('iptc_category', '')
                
                if not iptc_category:
                    raise ValueError("iptc_category parametresi gerekli")
                
                if export_treemap_tikz is None:
                    raise RuntimeError("IPTC treemap modülü yüklenemedi")
                
                print(f"[*] IPTC Treemap üretimi: {iptc_category}")
                
                # Tek IPTC kategorisi için treemap üret
                result = export_treemap_tikz(iptc_category)
                
                if not result.get('success'):
                    raise RuntimeError(result.get('error', 'Treemap üretimi başarısız'))
                
                print(f"[+] IPTC Treemap üretildi: {iptc_category}")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                
                response = json.dumps(result, ensure_ascii=False, indent=2)
                self.wfile.write(response.encode('utf-8'))
                
            except Exception as e:
                print(f" IPTC Treemap API Hatası: {e}")
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
        
        elif self.path == '/api/generate-all-iptc-treemaps':
            try:
                if generate_iptc_treemaps is None:
                    raise RuntimeError("IPTC treemap modülü yüklenemedi")
                
                print("[*] Tüm IPTC Treemap'leri üretimi başlatılıyor...")
                
                # Tüm IPTC kategorileri için treemap üret
                result = generate_iptc_treemaps()
                
                if not result.get('success'):
                    raise RuntimeError(result.get('error', 'Toplu treemap üretimi başarısız'))
                
                print(f"[+] Tüm IPTC Treemap'leri üretildi: {len(result.get('generated_files', []))} dosya")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                
                response = json.dumps(result, ensure_ascii=False, indent=2)
                self.wfile.write(response.encode('utf-8'))
                
            except Exception as e:
                print(f" Tüm IPTC Treemap'leri API Hatası: {e}")
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
║   Sunucu başlatıldı!                                         ║
║                                                                ║
║   Sayfayı açmak için: http://localhost:{PORT}                 ║
║   API Endpoint: POST http://localhost:{PORT}/api/analyze      ║
║                                                                ║
║   Dizin: {Path(__file__).parent}                             ║
║                                                                ║
║    Durdurmak için: Ctrl+C                                     ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")
        httpd.serve_forever()
        
except KeyboardInterrupt:
    print("\n\n Sunucu durduruldu.")
except OSError as e:
    print(f"\n Hata: {e}")
    if "Address already in use" in str(e):
        print(f"Port {PORT} zaten kullanımda. Lütfen başka bir program kapatın.")

