from flask import Flask, render_template, request, jsonify
from model_utils_groq import get_book_answer, is_osh_query

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_query = data.get("message", "")
    if not user_query or not is_osh_query(user_query):
        return jsonify({"answer": "I don't know.", "references": []})
    answer, refs = get_book_answer(user_query)
    return jsonify({"answer": answer, "references": refs})

# Temporary route for debugging: list all registered routes
@app.route("/__routes__", methods=["GET"])
def __routes__():
    lines = []
    for rule in app.url_map.iter_rules():
        methods = ",".join(sorted(rule.methods))
        lines.append(f"{rule.endpoint}: {rule.rule} [{methods}]")
    return "\n".join(lines)

# Temporary route to test OSH detection
@app.route("/__osh__", methods=["GET"])
def __osh__():
    q = request.args.get("q", "")
    return jsonify({"q": q, "is_osh": is_osh_query(q)})

# Debug print to verify route registration at import time
print("[app.py] Registered routes:", [rule.rule for rule in app.url_map.iter_rules()])

if __name__ == "__main__":
    # Avoid double-processing on startup: disable Flask's auto-reloader
    app.run(debug=True, use_reloader=False)