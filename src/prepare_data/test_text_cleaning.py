"""Regression tests for scripts/prepare_data/text_cleaning.py.

Run:  python -m scripts.prepare_data.test_text_cleaning
"""

from scripts.prepare_data.text_cleaning import clean, clean_entity

FAILURES = []


def check(name, got, want):
    if got != want:
        FAILURES.append(f"{name}\n   got : {got!r}\n   want: {want!r}")


# --- image lines are dropped entirely, caption included --------------------
check(
    "image line dropped",
    clean('ממוזער|left|325px|חתימתו של דילצדקי\nטקסט אמיתי כאן.', "v3"),
    "טקסט אמיתי כאן.",
)
check(
    "image line with placement prefix dropped",
    clean('שמאל|ממוזער|250px|אות שמע\nטקסט אמיתי כאן.', "v3"),
    "טקסט אמיתי כאן.",
)

# --- caption that spills onto the next line goes with it -------------------
check(
    "caption continuation dropped",
    clean('שמאל|ממוזער|250px|אות שמע\n[[פסקול]] בסרט\nטקסט אמיתי כאן.', "v3"),
    "טקסט אמיתי כאן.",
)
check(
    "long prose after an image is NOT treated as a continuation",
    clean('ממוזער|250px|כיתוב\nזהו משפט ארוך ואמיתי שאסור למחוק אותו בשום אופן שהוא.', "v3"),
    "זהו משפט ארוך ואמיתי שאסור למחוק אותו בשום אופן שהוא.",
)
check(
    "section header after an image is NOT treated as a continuation",
    clean('ממוזער|250px|כיתוב\n=== ניסוי ALPHA ===\nטקסט.', "v3"),
    "ניסוי ALPHA\nטקסט.",
)

# --- media variants found in the corpus during the 2026-08-12 audit --------
check(
    "Hebrew 'פיקסלים' size instead of px",
    clean('שגרירות סין בסטוקהולם|273x273 פיקסלים\nטקסט.', "v3"),
    "טקסט.",
)
check(
    "מסגרת (frame) directive",
    clean('שמאל|מסגרת|שכבת גבול סטוקס בתוך זורם צמיג.\nטקסט.', "v3"),
    "טקסט.",
)
check(
    "flag-icon template remnant",
    clean('קישור=דנמרק|טקסט=דנמרק|גבול|20px דנמרק\nטקסט.', "v3"),
    "טקסט.",
)
check("bare placement line", clean('right\nטקסט.', "v3"), "טקסט.")
check(
    "directive as a suffix, not a prefix",
    clean('דפנה דקל במהלך ביצוע השיר, 1992|ימין\nטקסט.', "v3"),
    "טקסט.",
)
check(
    "infobox parameter line",
    clean('|מסה= 2.9 (כל אחד)\n|גיל=\nטקסט.', "v3"),
    "טקסט.",
)
check(
    "table cell attribute line",
    clean('width="20px"|\nטקסט.', "v3"),
    "טקסט.",
)
check("leftover html tag stripped", clean('</small>*הנתונים מבוססים.', "v3"), "הנתונים מבוססים.")
check(
    "stray pipe becomes a space, surrounding text kept",
    clean('דיני ההתיישנות | ארז קמיניץ.', "v3"),
    "דיני ההתיישנות ארז קמיניץ.",
)

# --- article body glued onto an image caption must survive -----------------
_GLUED = ("ממוזער|סרטוט כללי של שסתום בטיחותשסתום בטיחות הוא שסתום המשחרר באופן "
          "אוטומטי את התווך הזורם במערכת אם הלחץ עולה על ערך שניקבע מראש כמסכן את "
          "המערכת. מוצא שסתום הבטיחות יכול להיות לאוויר או שפיכה חופשית או למערכת "
          "משנית המיועדת לטפל בחומר ששוחרר.")
_out = clean(_GLUED, "v3")
check("glued caption+body keeps the body", len(_out.split()) > 30, True)
check("glued caption+body drops the directive", "ממוזער|" in _out, False)

# --- size tokens: markup residue goes, real mentions stay ------------------
check("flag-icon size token at line start",
      clean('טקסט.\n30px טאיוואן - חוזה של 916 מיליון דולר\nעוד.', "v3"),
      "טקסט.\nטאיוואן - חוזה של 916 מיליון דולר\nעוד.")
check("size token with dash separator",
      clean('טקסט.\n20px - כרמית בוריאן\nעוד.', "v3"),
      "טקסט.\nכרמית בוריאן\nעוד.")
check("size mention mid-sentence must survive",
      clean("המסך מציג ברזולוציה של 1920x1080 פיקסלים והוא איכותי.", "v3"),
      "המסך מציג ברזולוציה של 1920x1080 פיקסלים והוא איכותי.")

# A 24-word body glued to a caption was deleted outright by the old 25-word
# keep-threshold. Regression guard for both the keep and the size-prefix strip.
_GLUED24 = ("ממוזער|271x271 פיקסליםאלנה גלנדאור מסרה עדות ביד ושם על מנת שיכירו "
            "בה כחסידת אומות העולם והיא אכן הוכרה ככזו בשנת אלפיים ושתיים.")
_o = clean(_GLUED24, "v3")
check("24-word glued body is not deleted", len(_o.split()) >= 18, True)
check("glued body loses its size prefix", _o.startswith("אלנה"), True)

# --- content headers keep their words but lose the markup ------------------
check(
    "content header unwrapped",
    clean('פתיח.\n== ביוגרפיה ==\nהמשך.', "v3"),
    "פתיח.\nביוגרפיה\nהמשך.",
)
check(
    "deeper header unwrapped",
    clean('א.\n=== ניסוי ALPHA ===\nב.', "v3"),
    "א.\nניסוי ALPHA\nב.",
)
check(
    "nbsp inside a content header becomes a plain space",
    clean('א.\n== תפוצה\xa0ומרחב ==\nב.', "v3"),
    "א.\nתפוצה ומרחב\nב.",
)

# --- footer cut, including the non-breaking-space form ---------------------
check(
    "spaced footer cut",
    clean('גוף המאמר.\n== קישורים חיצוניים ==\nקישור כלשהו', "v3"),
    "גוף המאמר.",
)
check(
    "nbsp inside the footer name still cuts",
    clean('גוף המאמר.\n== קישורים\xa0חיצוניים ==\nקישור כלשהו', "v3"),
    "גוף המאמר.",
)
check(
    "further-reading footer cut (missing from v1 and v2)",
    clean('גוף המאמר.\n== לקריאה נוספת ==\nספר כלשהו', "v3"),
    "גוף המאמר.",
)

# --- category tag: line-initial only ---------------------------------------
check(
    "line-initial category tag cuts",
    clean('גוף המאמר.\nקטגוריה:סרטים', "v3"),
    "גוף המאמר.",
)
check(
    "mid-sentence 'קטגוריה:' is prose and must survive",
    clean('שינוי הכללים של הקטגוריה: סרטים זכאים חייבים להיות ארוכים.', "v3"),
    "שינוי הכללים של הקטגוריה: סרטים זכאים חייבים להיות ארוכים.",
)

# --- links and bullets ------------------------------------------------------
check("piped link unwrapped", clean('ראו [[נפות טורקיה|נפת]] אלזי.', "v3"), "ראו נפת אלזי.")
check("plain link unwrapped", clean('ראו [[טורקיה]] היום.', "v3"), "ראו טורקיה היום.")
check("bullet stripped", clean('* מטה הבנק המרכזי', "v3"), "מטה הבנק המרכזי")

# --- older levels still reproduce their historical behaviour ---------------
check(
    "v1 misses the spaced header (the original bug)",
    clean('גוף.\n== קישורים חיצוניים ==\nקישור', "v1"),
    "גוף.\n== קישורים חיצוניים ==\nקישור",
)
check("v1 catches the exact header", clean('גוף.\n==ראו גם==\nעוד', "v1"), "גוף.")
check("v2 catches the spaced header", clean('גוף.\n== ראו גם ==\nעוד', "v2"), "גוף.")
check(
    "v2 does NOT drop image lines",
    clean('ממוזער|250px|כיתוב\nגוף.', "v2"),
    "ממוזער|250px|כיתוב\nגוף.",
)

# --- entity-field normalisation --------------------------------------------
check("entity: wiki link unwrapped",
      clean_entity("טוני בנט שר שירים של [[רוג'רס והארט]]"),
      "טוני בנט שר שירים של רוג'רס והארט")
check("entity: nbsp becomes a plain space", clean_entity('אה"מ\xa0מגניפיסנט'), 'אה"מ מגניפיסנט')
check("entity: pipe becomes a space", clean_entity("דנמרק|גבול"), "דנמרק גבול")
check("entity: clean value unchanged", clean_entity("ירושלים"), "ירושלים")

if FAILURES:
    print(f"FAILED {len(FAILURES)} check(s):\n")
    for f in FAILURES:
        print(f + "\n")
    raise SystemExit(1)
print("all text_cleaning checks passed")
