def render_funnel_diagram():
    """Returns the HTML/CSS for the Input -> Process -> Output funnel diagram,
    shown on the Dashboard page. Mirrors the classic three-panel funnel layout
    (input list / process funnel / output ranked list) using RE-TCP's real
    pipeline terminology and brand colors."""
    import textwrap

    input_items = [
        ("#BD582C", "REQ-01 \u2014 FR"),
        ("#E48312", "REQ-05 \u2014 NFR"),
        ("#13294B", "REQ-08 \u2014 FR"),
        ("#6B8E23", "REQ-03 \u2014 NFR"),
        ("#8A4D0A", "REQ-11 \u2014 FR"),
        ("#999999", "..."),
    ]
    output_items = [
        ("1", "TC1", "High"),
        ("2", "TC2", "High"),
        ("3", "TC3", "Medium"),
        ("4", "TC4", "Low"),
        ("5", "TC5", "Low"),
        ("6", "...", "Low"),
    ]

    def badge(level):
        color = {"High": "#c0392b", "Medium": "#b7950b", "Low": "#888888"}[level]
        bg = {"High": "#FBEAEA", "Medium": "#FBF3D9", "Low": "#F0F0F0"}[level]
        return f'<span style="background:{bg}; color:{color}; font-size:0.7rem; font-weight:700; padding:2px 10px; border-radius:10px;">{level}</span>'

    input_rows = "".join(f"""
        <div style="display:flex; align-items:center; gap:10px; background:#ffffff; border-radius:8px;
                    padding:8px 12px; margin-bottom:8px; border:1px solid #e3e3e3;">
            <div style="width:14px; height:14px; border-radius:4px; background:{color};"></div>
            <div style="font-size:0.8rem; color:#333; font-weight:600;">{label}</div>
        </div>
    """ for color, label in input_items)

    output_rows = "".join(f"""
        <div style="display:flex; align-items:center; justify-content:space-between; background:#ffffff;
                    border-radius:8px; padding:8px 12px; margin-bottom:8px; border:1px solid #e3e3e3;">
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:22px; height:22px; border-radius:50%; background:#13294B; color:#fff;
                            font-size:0.72rem; font-weight:700; display:flex; align-items:center;
                            justify-content:center;">{rank}</div>
                <div style="font-size:0.8rem; color:#333; font-weight:600;">{tc}</div>
            </div>
            {badge(level)}
        </div>
    """ for rank, tc, level in output_items)

    prep_steps = ["Tokenize", "Remove Stopwords", "Lemmatize"]
    prep_rows = "".join(f"""
        <div style="display:flex; align-items:center; gap:8px; background:#ffffff; border-radius:8px;
                    padding:7px 12px; margin-bottom:8px; border:1px solid #e3e3e3;">
            <div style="width:8px; height:8px; border-radius:50%; background:#6B8E23;"></div>
            <div style="font-size:0.78rem; color:#333; font-weight:700;">{step}</div>
        </div>
    """ for step in prep_steps)

    html = f"""
    <div style="display:flex; align-items:stretch; gap:0; margin-top:6px;">

        <div style="flex:0.85; background:#EAF2FB; border:1px solid #cfe0f3; border-radius:14px; padding:18px;">
            <div style="text-align:center; margin-bottom:10px;">
                <div style="width:52px; height:52px; border-radius:50%; background:#DAE8FC; margin:0 auto 8px auto;
                            display:flex; align-items:center; justify-content:center; font-size:1.4rem;">\U0001F4CB</div>
                <div style="font-weight:800; color:#13294B; font-size:1rem;">INPUT</div>
                <div style="font-size:0.78rem; color:#555;">Requirements</div>
            </div>
            {input_rows}
        </div>

        <div style="flex:0 0 44px; display:flex; align-items:center; justify-content:center; font-size:1.6rem; color:#13294B;">&#8594;</div>

        <div style="flex:0.7; background:#EAF7ED; border:1px solid #c9e6d3; border-radius:14px; padding:18px;">
            <div style="text-align:center; margin-bottom:10px;">
                <div style="width:52px; height:52px; border-radius:50%; background:#D5E8D4; margin:0 auto 8px auto;
                            display:flex; align-items:center; justify-content:center; font-size:1.4rem;">\U0001F9F9</div>
                <div style="font-weight:800; color:#13294B; font-size:1rem;">PREPROCESSING</div>
                <div style="font-size:0.78rem; color:#555;">Clean &amp; Normalize</div>
            </div>
            {prep_rows}
        </div>

        <div style="flex:0 0 44px; display:flex; align-items:center; justify-content:center; font-size:1.6rem; color:#13294B;">&#8594;</div>

        <div style="flex:0 0 220px; display:flex; flex-direction:column; align-items:center; justify-content:center;
                    border:2px dashed #b7a3d9; border-radius:50%; padding:18px; margin:0 4px;">
            <div style="width:54px; height:54px; border-radius:50%; background:#EDE4F7; display:flex; align-items:center;
                        justify-content:center; font-size:1.4rem; margin-bottom:6px;">\u2699\ufe0f</div>
            <div style="font-weight:800; color:#4a2e7a; font-size:0.95rem;">PROCESS</div>
            <div style="font-size:0.75rem; color:#4a2e7a; margin-bottom:8px; text-align:center;">Aspect-Based<br>Prioritization</div>
            <div style="font-size:1.4rem; margin:4px 0;">\U0001F53D</div>
            <div style="font-size:0.66rem; color:#666; text-align:center; margin-top:4px;">
                Classify, score, and rank<br>requirements &amp; test cases
            </div>
        </div>

        <div style="flex:0 0 44px; display:flex; align-items:center; justify-content:center; font-size:1.6rem; color:#13294B;">&#8594;</div>

        <div style="flex:0.85; background:#FBEFE3; border:1px solid #f0d3b0; border-radius:14px; padding:18px;">
            <div style="text-align:center; margin-bottom:10px;">
                <div style="width:52px; height:52px; border-radius:50%; background:#F6D2AE; margin:0 auto 8px auto;
                            display:flex; align-items:center; justify-content:center; font-size:1.4rem;">\U0001F4C8</div>
                <div style="font-weight:800; color:#13294B; font-size:1rem;">OUTPUT</div>
                <div style="font-size:0.78rem; color:#555;">Prioritized Test Cases</div>
            </div>
            {output_rows}
        </div>

    </div>
    """
    return "\n".join(line.lstrip() for line in html.split("\n"))
