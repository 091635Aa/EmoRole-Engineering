# -*- coding: utf-8 -*-
"""P1.7 GGUF Control Vector 线上验证(llama-server HTTP 版)
两个 llama-server 实例(带/不带 --control-vector)各答 3 个问题, 裁判评分对比。
"""
import os, sys, json, re, time, subprocess, urllib.request, urllib.error
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_sweep as rs
import p15_sweep as p15
import p17_reverse as p17

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA_SERVER = r"<workspace>/pykits/llama-cpp-bin\llama-server.exe"
MODEL = r"<workspace>/pykits/models\Qwen3-4B-Q4_K_M.gguf"
CVEC = os.path.join(HERE, "cvec_emocompanion_p17.gguf")
PORT = 8031
QUESTIONS = p15.build_testset()[:3]

def wait_up(timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if json.loads(r.read().decode()).get("status") == "ok":
                    return True
        except Exception:
            time.sleep(3)
    return False

def chat(q):
    body = {"messages": [{"role": "system", "content": p17.PERSONA},
                          {"role": "user", "content": q}],
            "max_tokens": 96, "temperature": 0.9, "top_p": 0.9, "top_k": 50, "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        j = json.loads(r.read().decode("utf-8"))
    return j["choices"][0]["message"]["content"].strip()

def run_mode(use_cvec):
    cmd = [LLAMA_SERVER, "-m", MODEL, "--port", str(PORT), "-ngl", "99", "-c", "2048"]
    if use_cvec:
        cmd += ["--control-vector", CVEC]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        if not wait_up():
            raise RuntimeError("server 启动超时")
        texts = []
        for q in QUESTIONS:
            t = chat(q)
            texts.append({"u": q, "text": t[:300],
                          "tone_hit": sum(t.count(w) for w in rs.TONE_WORDS)})
            print(f"  [mode] {q[:16]} → {t[:56]}... (词命中{texts[-1]['tone_hit']})", flush=True)
        return texts
    finally:
        proc.terminate()

def main():
    results = {}
    for mode, use_cvec in (("plain", False), ("cvec", True)):
        print(f"==== {mode} ====", flush=True)
        results[mode] = run_mode(use_cvec)
    print("[judge] load Qwen3-8B", flush=True)
    jm, jt = p15.load_judge()
    for mode in ("plain", "cvec"):
        fids = []
        for it in results[mode]:
            out = p15.ask(jm, jt, p15.JUDGE_FID.format(u=it["u"], t=it["text"]))
            m = re.search(r"\d{1,3}", out)
            fids.append(min(100, int(m.group())) if m else None)
        valid = [x for x in fids if x is not None]
        results[mode + "_fidelity"] = round(float(sum(valid) / len(valid)), 1) if valid else None
        print(f"  {mode} 保真 {results[mode + '_fidelity']}", flush=True)
    json.dump(results, open(os.path.join(HERE, "p17_cvec_test.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved: p17_cvec_test.json")

if __name__ == "__main__":
    main()
