import os
import pickle
from flask import Flask, render_template_string, request, send_from_directory
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Önbelleğe alınmış vektörleri ve yapay zeka modelini yükle
with open("vektorler.pkl", "rb") as f:
  data = pickle.load(f)
  df_kuran = data["df"]
  vektorler = data["vektorler"]

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


@app.route("/sw.js")
def serve_sw():
  return send_from_directory(
      os.path.join(app.root_path, "static"),
      "sw.js",
      mimetype="application/javascript",
  )


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no, viewport-fit=cover">
    <title>Kur'an Anlamsal Arama</title>
    
    <!-- PWA VE IOS GEREKSİNİMLERİ -->
    <link rel="manifest" href="/static/manifest.json">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="Kur'an Arama">
    <link rel="apple-touch-icon" href="/static/icon.png">

    <script src="https://jsdelivr.net"></script>
    <style>
        body { background-color: #f1f5f9; padding-top: env(safe-area-inset-top); }
    </style>
</head>
<body class="min-h-screen flex flex-col justify-between font-sans">

    <main class="w-full max-w-md mx-auto px-4 py-6 flex-grow">
        <div class="text-center my-6">
            <h1 class="text-3xl font-black text-emerald-700 tracking-tight">📖 Kur'an-ı Kerim</h1>
            <p class="text-slate-500 text-xs mt-1">Yyapay Zeka Destekli Anlamsal Arama Motoru</p>
        </div>

        <!-- ARAMA FORMU -->
        <form method="GET" action="/" class="mb-6">
            <div class="flex shadow-md rounded-2xl overflow-hidden border border-emerald-100 bg-white p-1">
                <input type="text" name="q" placeholder="Örn: merhamet, sabır, zorluk..." value="{{ query }}" required 
                       class="flex-1 px-4 py-3 text-sm focus:outline-none text-slate-800">
                <button type="submit" class="bg-emerald-600 text-white font-semibold text-sm px-5 rounded-xl hover:bg-emerald-700 active:scale-95 transition-all">
                    Ara
                </button>
            </div>
            
            <div class="mt-3 flex items-center justify-between px-1">
                <label class="text-xs font-semibold text-slate-500">Sonuç Sayısı:</label>
                <select name="limit" class="text-xs bg-white border border-slate-200 rounded-lg p-1 text-slate-700 focus:outline-none">
                    <option value="3" {% if limit == 3 %}selected{% endif %}>3 Sonuç</option>
                    <option value="5" {% if limit == 5 %}selected{% endif %}>5 Sonuç</option>
                    <option value="10" {% if limit == 10 %}selected{% endif %}>10 Sonuç</option>
                </select>
            </div>
        </form>

        <!-- SONUÇLAR -->
        <div class="space-y-4">
            {% if results %}
                {% for row in results %}
                <div class="bg-white p-5 rounded-2xl shadow-xs border border-slate-100 space-y-3">
                    <div class="flex justify-between items-center border-b border-slate-100 pb-2">
                        <span class="text-xs font-bold text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-full">
                            {{ row.sure }} - Ayet {{ row.ayet_no }}
                        </span>
                        <span class="text-[10px] font-mono text-slate-400">Skor: %{{ row.skor }}</span>
                    </div>
                    <p class="text-right font-serif text-lg text-slate-800 leading-loose dir-rtl" style="font-family: 'Amiri', serif;">
                        {{ row.arapca }}
                    </p>
                    <p class="text-sm text-slate-600 leading-relaxed border-t border-slate-50 pt-2 italic">
                        <strong>Meal:</strong> {{ row.meal }}
                    </p>
                </div>
                {% endfor %}
            {% elif query %}
                <p class="text-center text-slate-400 text-sm py-8">Eşleşen bir ayet bulunamadı.</p>
            {% endif %}
        </div>
    </main>

    <script>
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js').then(() => console.log('PWA Aktif!'));
        }
    </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def search():
  query = request.args.get("q", "").strip()
  limit = int(request.args.get("limit", 5))
  results = []

  if query:
    sorgu_vektoru = model.encode([query])
    benzerlikler = cosine_similarity(sorgu_vektoru, vektorler)[0]
    import numpy as np

    en_iyi_indexler = np.argsort(benzerlikler)[::-1][:limit]

    for idx in en_iyi_indexler:
      ayet = df_kuran.iloc[idx]
      results.append({
          "sure": ayet["sure"],
          "ayet_no": ayet["ayet_no"],
          "arapca": ayet["arapca"],
          "meal": ayet["meal"],
          "skor": int(benzerlikler[idx] * 100),
      })

  return render_template_string(
      HTML_TEMPLATE, results=results, query=query, limit=limit
  )


if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=5000)
