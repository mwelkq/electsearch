"""
Рисунки для розділу 3 курсової (PNG, 200 dpi, світла тема для друку).

  python tools/make_figures.py          # -> figures/*.png і figures/examples.md

Цифри беруться з файлів результатів у eval/ (крім кількох проміжних значень,
які в журналі зафіксовано лише підсумком — вони підписані в STAGES нижче).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# перші три слоти перевіреної категоріальної палітри (розрізняються і при дальтонізмі)
COLORS = {"bm25": "#2a78d6", "embed": "#eb6834", "hybrid": "#1baf7a"}
NAMES = {"bm25": "BM25", "embed": "Embeddings (e5-small)", "hybrid": "Гібрид (RRF)"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
METHODS = ["bm25", "embed", "hybrid"]

plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
})


def load(name):
    return json.load(open(EVAL / f"{name}.json", encoding="utf-8"))


def style_y(ax, top=1.0, label=None):
    ax.set_ylim(0, top)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
    if label:
        ax.set_ylabel(label)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print(OUT / name)


def bars_with_gap(ax, xs, heights, width, color, label=None):
    """Стовпці з тонким проміжком кольору фону між сусідніми (2 px при 200 dpi)."""
    return ax.bar(xs, heights, width, color=color, label=label, edgecolor=SURFACE, linewidth=1.2)


# 1. Порівняння методів пошуку (фінальна конфігурація)
def fig_methods():
    s = load("results_v3_e5small256_exp")["summary"]
    metrics = ["Hit@1", "Hit@3", "Hit@5", "MRR"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    w = 0.26
    for j, m in enumerate(METHODS):
        xs = [i + (j - 1) * w for i in range(len(metrics))]
        vals = [s[m]["all"][k] for k in metrics]
        bars_with_gap(ax, xs, vals, w, COLORS[m], NAMES[m])
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(metrics)), metrics)
    style_y(ax, 1.08, "Частка запитів / MRR")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.set_title("Якість пошуку сторінок: 52 запити, фінальна конфігурація", loc="left", fontsize=11.5)
    save(fig, "fig1_methods.png")


# 2. Як змінювалась якість пошуку (MRR) з покращеннями
STAGES = [  # (підпис, файл результатів)
    ("Індекс v1", "results_e5small256_exp"),
    ("Індекс v2\n(крихти)", "results_v2_e5small256_exp"),
    ("+ вага\nзаголовка", "results_v2tw1_e5small256_exp"),
    ("+ лема,\nсиноніми", "results_v3_e5small256_exp"),
]


def fig_progress():
    data = {m: [load(f)["summary"][m]["all"]["MRR"] for _, f in STAGES] for m in METHODS}
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    xs = range(len(STAGES))
    for m in METHODS:
        ax.plot(xs, data[m], color=COLORS[m], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=NAMES[m])
        ax.annotate(f"{data[m][-1]:.3f}", (len(STAGES) - 1, data[m][-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.set_xticks(list(xs), [s for s, _ in STAGES])
    ax.set_xlim(-0.3, len(STAGES) - 0.55)
    style_y(ax, 1.0, "MRR")
    ax.set_ylim(0.6, 0.9)
    ax.legend(loc="upper left")
    ax.set_title("Внесок покращень у якість пошуку (MRR, 52 запити)", loc="left", fontsize=11.5)
    save(fig, "fig2_progress.png")


# 3. Точність RAG за версіями
def rag_acc(tag):
    return load(f"rag_{tag}_summary")["summary"]["accuracy"]


RAG_STAGES = [
    ("v1, k=5", "rag_hybrid_lite"),
    ("v2, k=5", "rag_hybrid_lite_v2"),
    ("+ вага\nзаголовка", "rag_hybrid_lite_v2tw1"),
    ("+ лема, синоніми,\nпромпт, k=5", "rag_hybrid_lite_v3_k5"),
    ("k=8", "rag_hybrid_lite_v3_k8"),
]


def fig_rag(extra=None):
    stages = RAG_STAGES + (extra or [])
    vals = [rag_acc(t) for _, t in stages]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    xs = range(len(stages))
    bars_with_gap(ax, xs, vals, 0.55, COLORS["hybrid"])
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(list(xs), [s for s, _ in stages], fontsize=9)
    style_y(ax, 1.08, "Частка правильних сторінок")
    ax.set_title("Точність RAG (гібридний пошук + gemini-3.1-flash-lite, 52 запити)", loc="left", fontsize=11.5)
    save(fig, "fig3_rag_accuracy.png")


# 4. Hit@3 за категоріями запитів
CATS = {"nav_direct": "Навігаційні\n(пряма назва)", "nav_colloquial": "Навігаційні\n(розмовні)",
        "info": "Інформаційні", "abbrev": "Абревіатури", "person": "Персони",
        "course": "Дисципліни", "colloquial": "Розмовні /\nз помилками"}


def fig_categories(name="results_e5small256_exp", out="fig4_categories_v1.png", title=""):
    s = load(name)["summary"]
    cats = [c for c in CATS if c in s["bm25"]]
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    w = 0.26
    for j, m in enumerate(METHODS):
        xs = [i + (j - 1) * w for i in range(len(cats))]
        bars_with_gap(ax, xs, [s[m][c]["Hit@3"] for c in cats], w, COLORS[m], NAMES[m])
    ax.set_xticks(range(len(cats)), [f"{CATS[c]}\n(n={s['bm25'][c]['n']})" for c in cats], fontsize=8.8)
    style_y(ax, 1.05, "Hit@3")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    ax.set_title(title, loc="left", fontsize=11.5)
    save(fig, out)


# 5. Гіпотеза H4: лематизація
def fig_h4():
    lem = load("results_v3_e5small256_exp")["summary"]
    nol = load("results_h4_nolemma_main")["summary"]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    groups = [("BM25", "bm25"), ("Гібрид", "hybrid")]
    w = 0.34
    for j, (lab, src, col) in enumerate([("без лематизації", nol, "#b8b6ae"), ("з лематизацією", lem, COLORS["bm25"])]):
        xs = [i + (j - 0.5) * w for i in range(len(groups))]
        vals = [src[m]["all"]["MRR"] for _, m in groups]
        bars_with_gap(ax, xs, vals, w, col, lab)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(range(len(groups)), [g for g, _ in groups])
    style_y(ax, 1.0, "MRR")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.set_title("Гіпотеза H4: вплив лематизації (52 запити)", loc="left", fontsize=11.5)
    save(fig, "fig5_h4_lemma.png")


# 6. Робочий набір проти нових (незалежних) запитів
def fig_generalization():
    main = load("results_v3_e5small256_exp")["summary"]["hybrid"]["all"]
    user = load("results_user_final")["summary"]["hybrid"]["all"]
    rag_main = load("rag_rag_v4_gemini-3.1-flash-lite_summary")["summary"]["accuracy"]
    rag_user = load("rag_rag_user_final_summary")["summary"]["accuracy"]
    labels = ["Пошук: Hit@1", "Пошук: Hit@5", "Пошук: MRR", "RAG: точність"]
    series = [(f"Робочий набір (n={main['n']}), на ньому налаштовували", "#2a78d6",
               [main["Hit@1"], main["Hit@5"], main["MRR"], rag_main]),
              (f"Нові запити (n={user['n']}), система їх не бачила", "#eb6834",
               [user["Hit@1"], user["Hit@5"], user["MRR"], rag_user])]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    w = 0.36
    for j, (lab, col, vals) in enumerate(series):
        xs = [i + (j - 0.5) * w for i in range(len(labels))]
        bars_with_gap(ax, xs, vals, w, col, lab)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(range(len(labels)), labels)
    style_y(ax, 1.1, "Частка запитів / MRR")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)
    ax.set_title("Гібридний пошук і RAG: робочий набір проти нових запитів", loc="left", fontsize=11.5)
    save(fig, "fig6_generalization.png")


# Таблиця прикладів: де виграє кожен метод
def examples():
    rows = load("results_e5small256_exp")["rows"]
    b = {x["id"]: x for x in rows["bm25"]}
    e = {x["id"]: x for x in rows["embed"]}
    r = lambda x: str(x) if x else "—"
    lines = ["# Приклади: де виграє BM25, а де embeddings", "",
             "Ранг правильної сторінки (1 = перше місце, «—» = немає в топ-10). Конфігурація експерименту 1",
             "(індекс v1, e5-small, без ваги заголовка), тобто «чисті» методи до покращень.", "",
             "| Запит | Категорія | BM25 | Embeddings | Чому |", "|---|---|---|---|---|"]
    why = {
        "q014": "на сторінці немає слова «очолює» — embeddings розуміють зміст (декан)",
        "q026": "«деканат» / «телефон» ≠ словам сторінки «Контакти» — потрібна семантика",
        "q025": "у BM25 «факультет» є майже всюди; embeddings бачать «адресу» як зміст",
        "q013": "перефразування: «заплатити» vs «оплата»",
        "q035": "«КН» не збігається з «Комп’ютерні науки»; embeddings частково ловлять зв’язок",
        "q038": "точний шифр «122» і назва — сильна сторона BM25",
    }
    for i in ["q014", "q026", "q025", "q013", "q035", "q038"]:
        lines.append(f"| {b[i]['query']} | {b[i]['category']} | {r(b[i]['first_rank'])} | {r(e[i]['first_rank'])} | {why[i]} |")
    v3 = load("results_v3_e5small256_exp")["rows"]
    b3 = {x["id"]: x for x in v3["bm25"]}
    e3 = {x["id"]: x for x in v3["embed"]}
    lines += ["", "Після покращень BM25 (вага заголовка, лематизація, синоніми) — фінальна конфігурація:", "",
              "| Запит | Категорія | BM25 | Embeddings | Чому |", "|---|---|---|---|---|"]
    why3 = {
        "q004": "сторінка-хаб «Лабораторії»: текст — лише перелік, допомагає збіг із заголовком",
        "q005": "те саме для «Кафедри»",
        "q037": "абревіатура ІСТ → розшифровка дає точний лексичний збіг",
        "q020": "власна назва «Startup Jam» — точний збіг",
        "q014": "embeddings і далі кращі на перефразуваннях",
        "q026": "і на синонімічних запитах",
    }
    for i in ["q004", "q005", "q037", "q020", "q014", "q026"]:
        lines.append(f"| {b3[i]['query']} | {b3[i]['category']} | {r(b3[i]['first_rank'])} | {r(e3[i]['first_rank'])} | {why3[i]} |")
    lines += ["", "Висновок: методи помиляються на **різних** запитах, тому гібрид (RRF) виграє, коли обидва достатньо сильні."]
    (OUT / "examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "examples.md")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    fig_methods()
    fig_progress()
    fig_rag()
    fig_categories("results_e5small256_exp", "fig4_categories_v1.png",
                   "Hit@3 за категоріями запитів: «чисті» методи (індекс v1)")
    fig_categories("results_v3_e5small256_exp", "fig4b_categories_final.png",
                   "Hit@3 за категоріями запитів: фінальна конфігурація")
    fig_h4()
    fig_generalization()
    examples()
