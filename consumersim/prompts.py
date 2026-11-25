"""Prompt rendering for the forecast step (one persona, one interview day)."""
import os
import datetime as dt
import hashlib
import json

SYSTEM = ("You are the U.S. resident described below, taking part in a telephone interview for the University of "
          "Michigan Surveys of Consumers. Answer according to this person's situation and views. Do not say that you are an AI.")

QUESTIONS = {
    'PAGO': "We are interested in how people are getting along financially these days. Would you say that you and your "
            "family living there are better off or worse off financially than you were a year ago?",
    'PEXP': "Now looking ahead--do you think that a year from now you and your family living there will be better off "
            "financially, or worse off, or just about the same as now?",
    'BUS12': "Now turning to business conditions in the country as a whole--do you think that during the next 12 months "
             "we'll have good times financially, or bad times, or what?",
    'BUS5': "Looking ahead, which would you say is more likely--that in the country as a whole we'll have continuous good "
            "times during the next 5 years or so, or that we will have periods of widespread unemployment or depression, or what?",
    'DUR': "About the big things people buy for their homes--such as furniture, a refrigerator, stove, television, and "
           "things like that. Generally speaking, do you think now is a good or a bad time for people to buy major household items?",
}
LABELS = {  # state -> label
    'PAGO': {'F': 'Better off', 'N': 'About the same', 'U': 'Worse off', 'D': "Don't know"},
    'PEXP': {'F': 'Better off', 'N': 'About the same', 'U': 'Worse off', 'D': "Don't know"},
    'BUS12': {'F': 'Good times', 'N': 'Good in some ways, bad in others', 'U': 'Bad times', 'D': "Don't know"},
    'BUS5': {'F': 'Continuous good times', 'N': 'Good in some ways, bad in others',
             'U': 'Periods of widespread unemployment or depression', 'D': "Don't know"},
    'DUR': {'F': 'Good time', 'N': 'Pros and cons, it depends', 'U': 'Bad time', 'D': "Don't know"},
}
MENU_ORDER = {'PAGO': 'FUND', 'PEXP': 'FUND', 'BUS12': 'FUND', 'BUS5': 'FUND', 'DUR': 'FUND'}
LABEL_TO_STATE = {q: {v.lower(): k for k, v in m.items()} for q, m in LABELS.items()}
SAME = 'Same as last time'
ADJACENT = {'F': 'ND', 'N': 'FUD', 'U': 'ND', 'D': 'FNU'}  # bounded update: at most one step per month
Q = list(QUESTIONS)

REGION = {1: 'the West', 2: 'the Midwest', 3: 'the Northeast', 4: 'the South'}
EDUC = {1: 'did not finish high school', 2: 'did not finish high school', 3: 'a high school diploma',
        4: 'some college but no degree', 5: 'a college degree', 6: 'a graduate degree'}
INCOME = {1: 'the bottom third', 2: 'the middle third', 3: 'the top third'}
PARTY = {1: 'a Republican', 2: 'a Democrat', 3: 'an independent'}


def _president_line_v1(day):
    if day < dt.date(2016, 11, 9):
        return 'The president is Barack Obama, a Democrat.'
    if day < dt.date(2017, 1, 20):
        return 'The president is Barack Obama, a Democrat. Donald Trump, a Republican, has been elected to succeed him.'
    if day < dt.date(2020, 11, 8):
        return 'The president is Donald Trump, a Republican.'
    if day < dt.date(2021, 1, 20):
        return 'The president is Donald Trump, a Republican. Joe Biden, a Democrat, has been elected to succeed him.'
    if day < dt.date(2024, 11, 7):
        return 'The president is Joe Biden, a Democrat.'
    if day < dt.date(2025, 1, 20):
        return 'The president is Joe Biden, a Democrat. Donald Trump, a Republican, has been elected to succeed him.'
    return 'The president is Donald Trump, a Republican.'


def president_line(day):
    if os.environ.get('T56_SNAPFIX') != '1':   # T56: the one-day-ahead version below was tested and not adopted
        return _president_line_v1(day)
    day = day - dt.timedelta(days=1)   # as known the day before the interview, as in env.political_text
    if day < dt.date(2016, 11, 9):
        return 'The president is Barack Obama, a Democrat.'
    if day < dt.date(2017, 1, 20):
        return 'The president is Barack Obama, a Democrat. Donald Trump, a Republican, has been elected to succeed him.'
    if day < dt.date(2020, 11, 7):
        return 'The president is Donald Trump, a Republican.'
    if day < dt.date(2021, 1, 20):
        return 'The president is Donald Trump, a Republican. Joe Biden, a Democrat, has been elected to succeed him.'
    # T43 fix: the 2024 election (race called 2024-11-06) and Trump's second term were missing (prompts said Biden)
    if day < dt.date(2024, 11, 6):
        return 'The president is Joe Biden, a Democrat.'
    if day < dt.date(2025, 1, 20):
        return 'The president is Joe Biden, a Democrat. Donald Trump, a Republican, has been elected to succeed him.'
    return 'The president is Donald Trump, a Republican.'


def president_then(month):
    d = dt.date(month // 100, month % 100, 15)
    if d < dt.date(2016, 11, 9):
        return 'Barack Obama, a Democrat, was president'
    if d < dt.date(2017, 1, 20):
        return 'Barack Obama, a Democrat, was president and Donald Trump, a Republican, had been elected to succeed him'
    if d < dt.date(2020, 11, 8):
        return 'Donald Trump, a Republican, was president'
    if d < dt.date(2021, 1, 20):
        return 'Donald Trump, a Republican, was president and Joe Biden, a Democrat, had been elected to succeed him'
    if d < dt.date(2024, 11, 7):
        return 'Joe Biden, a Democrat, was president'
    if d < dt.date(2025, 1, 20):
        return 'Joe Biden, a Democrat, was president and Donald Trump, a Republican, had been elected to succeed him'
    return 'Donald Trump, a Republican, was president'


def _pct(v):
    return v == v and v is not None and 0 <= v <= 100


def history_text(att, interview_month, month):
    """The persona's own answers to other questions in their earlier (t-12 or older) interview."""
    if not att:
        return ''
    lines = []
    g = att.get('GOVT')
    if g in (1, 3, 5):
        lines.append('the government was doing ' + {1: 'a good job', 3: 'only a fair job', 5: 'a poor job'}[g]
                     + ' with its economic policy (fighting inflation and unemployment)')
    b = att.get('BEXP')
    if b in (1, 3, 5):
        lines.append('business conditions in the country would be ' + {1: 'better', 3: 'about the same', 5: 'worse'}[b] + ' a year later')
    u = att.get('UNEMP')
    if u in (1, 3, 5):
        lines.append({1: 'there would be more unemployment', 3: 'unemployment would stay about the same',
                      5: 'there would be less unemployment'}[u] + ' in the coming year')
    px = att.get('PX1')
    if px == px and px is not None and -95 <= px <= 95:
        lines.append('prices in general would ' + (f'go up about {int(round(px))}%' if px > 0 else
                                                   'stay about where they were' if px == 0 else f'go down about {int(round(-px))}%')
                     + ' over the next 12 months')
    for col, what in [('HOM', 'buy a house'), ('CAR', 'buy a vehicle')]:
        v = att.get(col)
        if v in (1, 3, 5):
            lines.append({1: f'it was a good time to {what}', 3: f'it was neither a clearly good nor a clearly bad time to {what}',
                          5: f'it was a bad time to {what}'}[v])
    e = att.get('PEXP5')
    if e in (1, 3, 5):
        lines.append('five years later you would be ' + {1: 'better off financially', 3: 'about the same financially',
                                                         5: 'worse off financially'}[e])
    if _pct(att.get('PINC')):
        lines.append(f"the chance that your income would rise faster than prices over the next five years was about {int(round(att['PINC']))}%")
    if _pct(att.get('PJOB')):
        lines.append(f"the chance that you (or your spouse) would lose a job you wanted to keep in the next five years was about {int(round(att['PJOB']))}%")
    if not lines:
        return ''
    gap = (month // 100 * 12 + month % 100) - (interview_month // 100 * 12 + interview_month % 100)
    ago = 'about a year ago' if gap <= 18 else 'about two years ago'
    return (f'When you were interviewed for this survey {ago} ({president_then(interview_month)}), you said:\n'
            + '\n'.join(f'- {x}' for x in lines))


def facts_text(p, day):
    bits = []
    if p.get('AGE') == p.get('AGE') and p.get('AGE'):
        bits.append(f"You are {int(p['AGE'])} years old")
    if p.get('SEX') in (1, 2):
        bits.append('a man' if p['SEX'] == 1 else 'a woman')
    s = 'About you: ' + ', '.join(bits) + '.' if bits else 'About you:'
    more = []
    if p.get('REGION') in REGION:
        more.append(f"You live in {REGION[p['REGION']]}.")
    if p.get('EDUC') in EDUC:
        more.append(f"Your education: {EDUC[p['EDUC']]}.")
    if p.get('YTL3') in INCOME:
        more.append(f"Your household income is in {INCOME[p['YTL3']]} of U.S. households.")
    if p.get('HOMEOWN') in (1, 2):
        more.append('You own your home.' if p['HOMEOWN'] == 1 else 'You do not own your home.')
    if p.get('INVEST') in (1, 5):
        more.append('Your family has investments in the stock market.' if p['INVEST'] == 1
                    else 'Your family has no investments in the stock market.')
    if p.get('POLAFF') in PARTY:
        more.append(f"Politically, you usually think of yourself as {PARTY[p['POLAFF']]}.")
    more.append(president_line(day))
    return s + ' ' + ' '.join(more)


def render(persona, day, arm_cfg, info):
    """persona: dict with S_<q> states and fact fields; info: dict with optional 'macro', 'news', 'facts' text."""
    parts = []
    if arm_cfg.get('date'):
        parts.append(f"Today is {day.strftime('%B')} {day.day}, {day.year}.")
    if info.get('facts'):
        parts.append(info['facts'])
    prev = '\n'.join(f"{q}: {LABELS[q][persona['S_' + q]]}" for q in Q)
    parts.append(f"You were last interviewed about a month ago. In that interview you answered:\n{prev}")
    if info.get('memory'):
        parts.append(info['memory'])
    blocks = [info[k] for k in ('macro', 'news') if info.get(k)]
    if blocks:
        parts.append(('Since that interview:\n\n' if info.get('memory') else 'Since then:\n\n') + '\n\n'.join(blocks))
    style = arm_cfg.get('style', 'reask')
    if style == 'panel':
        parts.append("People's answers to these questions often stay the same from one month to the next, and sometimes "
                     "change when their circumstances or their view of the economy changes.")
    if style == 'keep':
        parts.append('Please answer the following questions according to what you think now. For each question, choose '
                     f'"{SAME}" if your answer has not changed, or choose your new answer.')
    else:
        parts.append('Please answer the following questions according to what you think now.')
    for q in Q:
        menu = [LABELS[q][s] for s in MENU_ORDER[q]]
        if style == 'keep':
            cur = persona['S_' + q]
            allowed = ADJACENT[cur] if arm_cfg.get('adjacent') else 'FUND'
            menu = [SAME] + [m for s, m in zip(MENU_ORDER[q], menu) if s != cur and s in allowed]
        parts.append(f"{q}\n{QUESTIONS[q]}\nResponse labels: {json.dumps(menu)}")
    if arm_cfg.get('cot'):
        parts.append('Before answering, think briefly as this person: how, if at all, do the things that happened since your '
                     'last interview affect your own household (your job and income, the prices you pay, your savings) and '
                     'your view of the country, and does any of it give you a reason to change each answer? '
                     'Return only a JSON object whose first key is "thinking" (two to four sentences), followed by the keys '
                     '"PAGO", "PEXP", "BUS12", "BUS5", "DUR", each set to the exact text of one listed response label.')
    else:
        parts.append('Return only a JSON object with exactly the keys "PAGO", "PEXP", "BUS12", "BUS5", "DUR", '
                     'each set to the exact text of one listed response label.')
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': '\n\n'.join(parts)}]


def parse(text, prev=None):
    """Return ({q: state}, error) from a model reply. prev: previous states (needed for the SAME label)."""
    if not text:
        return None, 'empty'
    s = text.strip()
    if s.startswith('```'):
        s = s.strip('`')
        s = s[s.find('{'):]
    i, j = s.find('{'), s.rfind('}')
    if i < 0 or j < 0:
        return None, 'no_json'
    try:
        obj = json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None, 'bad_json'
    out = {}
    for q in Q:
        v = obj.get(q)
        if isinstance(v, dict):
            v = v.get('choice')
        if not isinstance(v, str):
            return None, f'missing_{q}'
        if prev is not None and v.strip().strip('.').lower() == SAME.lower():
            out[q] = prev[q]
            continue
        st = LABEL_TO_STATE[q].get(v.strip().strip('.').lower())
        if st is None:
            return None, f'illegal_{q}'
        out[q] = st
    return out, None


def template_hash():
    src = json.dumps([SYSTEM, QUESTIONS, LABELS, MENU_ORDER], sort_keys=True)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- contrast elicitation (T12)
TOPICS = {
    'PAGO': 'How you and your family are getting along financially these days',
    'PEXP': "Your family's financial prospects for the year ahead",
    'BUS12': 'Business conditions in the country as a whole over the next 12 months',
    'BUS5': "The country's economic prospects over the next five years or so",
    'DUR': 'Whether now is a good time for people to buy major household items',
}


TOPICS_V2 = {
    'PAGO': 'How you and your family are getting along financially these days',
    'PEXP': 'Whether you and your family will be better off or worse off financially a year from now',
    'BUS12': 'Whether the country as a whole will have good times or bad times financially during the next 12 months',
    'BUS5': 'Whether the country will have continuous good times, or periods of widespread unemployment or depression, '
            'during the next five years or so',
    'DUR': 'Whether now is a good or bad time for people to buy major household items such as furniture, a refrigerator, '
           "a stove or a television (think about prices, interest rates and people's finances)",
}


OWN_VIEW = ('Judge as this particular person would, from your own point of view: think about the prices you pay, your job '
            'and income, any savings or investments you have, and your own political views, rather than about how the '
            'headlines sound.')


COUNTRY_TOPIC = 'The way things are going in the country, including how the people running the country are handling things'
SALIENCE = ('Most people do not follow economic statistics closely. Judge as this particular person would: by what they '
            'would actually notice in these snapshots and how it would make them feel.')


PEERS = ('Think about the people this person knows: family, friends, coworkers and neighbors. Answer for them rather '
         'than for this person. For each topic below, say in which snapshot most of those people would feel better about '
         'that topic ("A", "B", or "Same"), thinking of their own families for the personal topics.')


def render_contrast(facts, snap_a, snap_b, ask_change=True, topics=None, own_view=False, size=False, identity=False,
                    mood=False, salience=False, peers=False, perspective=False):
    topics = topics or TOPICS
    if mood:
        topics = {'COUNTRY': COUNTRY_TOPIC, **topics}
    parts = []
    if facts:
        parts.append(facts)
    parts.append('Below are two snapshots of the economic figures and front-page news an ordinary person '
                 'might have seen at two different times about a month apart. They are labeled A and B in no '
                 'particular order.')
    parts.append('SNAPSHOT A\n' + snap_a)
    parts.append('SNAPSHOT B\n' + snap_b)
    if own_view:
        parts.append(OWN_VIEW)
    if salience:
        parts.append(SALIENCE)
    if identity:
        parts.append("People's views about the economy can depend on more than the economic figures, including how they "
                     'feel about who is running the country. Answer as this particular person would.')
    parts.append('Compare the two snapshots from your own point of view. For each topic below, say which snapshot '
                 'looks better for that topic ("A", "B", or "Same").'
                 + (' Also say whether the difference is big enough that, as this person, you would give a '
                    'different answer to a survey question about that topic ("change": true or false).' if ask_change else ''))
    if perspective:   # T43 identity-forward question: which snapshot THIS person feels better about
        parts[-1] = PERSPECTIVE + (' Also say whether the difference is big enough that, as this person, you would give a '
                                   'different answer to a survey question about that topic ("change": true or false).' if ask_change else '')
    if peers:
        parts[-1] = PEERS + (' Also say whether the difference is big enough that most of them would give a different '
                             'answer to a survey question about that topic ("change": true or false).' if ask_change else '')
    parts.append('\n'.join(f'{q}: {t}' for q, t in topics.items()))
    keys = ', '.join(f'"{k}"' for k in topics)
    if size:
        parts[-2] = parts[-2].split(' Also say')[0] + (' Also say how big the difference is for you on that topic: '
                                                      '"large", "small", or "none".')
        parts.append(f'Return only a JSON object with exactly the keys {keys}. '
                     'Each value is an object {"better": "A" or "B" or "Same", "size": "large" or "small" or "none"}.')
    elif ask_change:
        parts.append(f'Return only a JSON object with exactly the keys {keys}. '
                     'Each value is an object {"better": "A" or "B" or "Same", "change": true or false}.')
    else:
        parts.append('Return only a JSON object with exactly the keys "PAGO", "PEXP", "BUS12", "BUS5", "DUR", '
                     'each set to "A", "B", or "Same".')
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': '\n\n'.join(parts)}]


def parse_contrast(text, now_label):
    """Return ({q: {'dir': +1/-1/0 for this month better/worse/same, 'change': bool}}, error)."""
    if not text:
        return None, 'empty'
    s = text.strip()
    i, j = s.find('{'), s.rfind('}')
    if i < 0 or j < 0:
        return None, 'no_json'
    try:
        obj = json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None, 'bad_json'
    out = {}
    for q in (['COUNTRY'] if 'COUNTRY' in obj else []) + list(Q):
        v = obj.get(q)
        ch = None
        size = None
        if isinstance(v, dict):
            ch = v.get('change')
            size = v.get('size')
            v = v.get('better')
        if not isinstance(v, str):
            return None, f'missing_{q}'
        v = v.strip().strip('.').upper()
        if v in ('SAME', 'ABOUT THE SAME', 'NEITHER'):
            d = 0
        elif v in ('A', 'B'):
            d = 1 if v == now_label else -1
        else:
            return None, f'illegal_{q}'
        if isinstance(ch, str):
            ch = ch.strip().lower() == 'true'
        if isinstance(size, str):
            size = {'large': 2, 'big': 2, 'small': 1, 'none': 0}.get(size.strip().lower())
        if size == 0:
            d = 0
        out[q] = {'dir': d, 'change': bool(ch) if ch is not None else None, 'size': size}
    return out, None


# ---------------------------------------------------------------- T43 identity-forward persona (registered 2026-11-18)
PERSPECTIVE = ('People in different situations, and with different political views, often react quite differently to the same '
               'news. Compare the two snapshots from your own point of view: think about what the differences mean for your own '
               "household's finances and how you feel about the way the country is being run. For each topic below, say which "
               'snapshot you feel better about ("A", "B", or "Same").')
LENS = {
    1: ('Like many Republicans, you see the economy partly through a partisan lens: when a Republican is president you tend to '
        'give the administration the benefit of the doubt and feel more confident, and when a Democrat is president you tend '
        'to be more skeptical and more worried about where the economy is headed.'),
    2: ('Like many Democrats, you see the economy partly through a partisan lens: when a Democrat is president you tend to '
        'give the administration the benefit of the doubt and feel more confident, and when a Republican is president you tend '
        'to be more skeptical and more worried about where the economy is headed.'),
    3: "You don't strongly identify with either party, so who is in charge matters less to you than how things are actually going.",
}


def lens_text(p):
    return LENS.get(p.get('POLAFF'), '')


def stakes_text(p):
    """What the persona's own situation exposes them to (facts about the household, no new figures)."""
    out = []
    if p.get('INVEST') == 1:
        out.append("Because your family owns stocks, swings in the stock market directly affect your household's wealth.")
    elif p.get('INVEST') == 5:
        out.append('Your family owns no stocks, so the stock market does not directly affect your finances.')
    if p.get('HOMEOWN') == 1:
        out.append('As a homeowner, you care about home values and mortgage rates.')
    elif p.get('HOMEOWN') == 2:
        out.append('As a renter, you feel rising rents and everyday prices directly.')
    if p.get('YTL3') == 1:
        out.append('With a household income in the bottom third, higher prices for food, gas and rent squeeze your budget '
                   'hard, and losing a job would be a serious blow.')
    elif p.get('YTL3') == 3:
        out.append('With a household income in the top third, you have some cushion against higher prices.')
    return ' '.join(out)


MARRY_TXT = {1: 'You are married or living with a partner.', 3: 'You are divorced or separated.', 4: 'You are widowed.',
             5: 'You have never been married.'}
INCOME_EXPOSURE = {1: 'With a household income in the bottom third, higher prices for essentials like food, gas and rent squeeze your budget hard.',
                   2: 'With a household income in the middle third, you notice higher prices for essentials but can usually absorb them.',
                   3: 'With a household income in the top third, you have some cushion against higher prices.'}


def family_text(p):
    out = [MARRY_TXT[p['MARRY']]] if p.get('MARRY') in MARRY_TXT else []
    k = p.get('NUMKID')
    if k == k and k is not None:
        k = int(k)
        out.append('You have no children under 18 at home.' if k == 0 else f"You have {k} child{'ren' if k > 1 else ''} under 18 at home.")
    return ' '.join(out)


def exposure_profile(p, exposed):
    """T48 uniform exposure layer: one sentence for every attribute that maps to an economic channel (identity, wealth,
    work, budget, housing, place). Sex and family carry no channel and appear as facts only."""
    out = [LENS[p['POLAFF']]] if p.get('POLAFF') in LENS else []
    if p.get('INVEST') == 1:
        out.append("Because your family owns stocks, swings in the stock market directly affect your household's wealth.")
    elif p.get('INVEST') == 5:
        out.append('Your family owns no stocks, so the stock market does not directly affect your finances.')
    age = p.get('AGE')
    if age == age and age is not None and age >= 65:
        out.append('At your age you are likely retired, so your income depends more on savings, pensions and Social Security than on a paycheck.')
    elif exposed:
        out.append('People in jobs like yours are often among the first to be laid off when the economy weakens.')
    elif age == age and age is not None:
        out.append('Jobs like yours are usually less exposed to layoffs when the economy weakens.')
    if p.get('YTL3') in INCOME_EXPOSURE:
        out.append(INCOME_EXPOSURE[p['YTL3']])
    if p.get('HOMEOWN') == 1:
        out.append('As a homeowner, home values and mortgage rates matter to you.')
    elif p.get('HOMEOWN') == 2:
        out.append('As a renter, rents and everyday prices hit you directly.')
    if p.get('REGION') in REGION:
        out.append(f"The prices and jobs that matter to you are those in {REGION[p['REGION']]}, where you live.")
    return ' '.join(out)
