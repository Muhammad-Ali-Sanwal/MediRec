"""
PDF Generator for MediRec Medicine Recommendation System.
Generates sleek, professional, non-editable PDF Medical Consultation & Prescription Reports using ReportLab.
"""
import io
import datetime
from typing import Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY


def generate_pdf_report(session: Dict[str, Any], recommendation: Dict[str, Any], recommender_details: Dict[str, Any]) -> bytes:
    """
    Generates a high-quality, professional medical PDF report and returns raw bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Premium Color Palette
    primary_color = colors.HexColor("#0f172a")     # Deep Slate
    accent_color = colors.HexColor("#0284c7")      # Medical Sky Blue
    secondary_color = colors.HexColor("#334155")   # Dark Slate
    light_bg = colors.HexColor("#f8fafc")          # Soft Light Grey
    alert_red = colors.HexColor("#dc2626")         # Clinical Warning Red
    alert_bg = colors.HexColor("#fef2f2")          # Light Red Tint

    # Typography
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.white,
        alignment=TA_LEFT
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#e2e8f0"),
        alignment=TA_LEFT
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=accent_color,
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=secondary_color
    )

    bold_body_style = ParagraphStyle(
        "BoldBodyText",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=TA_CENTER
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1e293b")
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell_style,
        fontName="Helvetica-Bold"
    )

    story = []

    # 1. Header Banner
    now_str = datetime.datetime.now().strftime("%B %d, %Y - %H:%M PKT")
    header_data = [
        [
            Paragraph("<b>MediRec AI Clinical Assistant</b>", title_style),
            Paragraph(f"<b>Date:</b> {now_str}<br/><b>Ref ID:</b> MDR-{datetime.datetime.now().strftime('%Y%m%d%H%M')}", subtitle_style)
        ],
        [
            Paragraph("Official Medical Consultation & Prescription Report", subtitle_style),
            Paragraph("<b>Course:</b> CS619 FYP | <b>Dev:</b> M. Ali Sanwal (BC240440384)", subtitle_style)
        ]
    ]

    header_table = Table(header_data, colWidths=[340, 200])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), primary_color),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 10))

    # 2. Patient Presentation Section
    story.append(Paragraph("1. PATIENT PROFILE & CLINICAL PRESENTATION", section_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=accent_color, spaceBefore=2, spaceAfter=6))

    age = session.get("age", "N/A")
    gender = session.get("gender", "N/A")
    severity_map = {1: "Mild (Level 1)", 2: "Moderate (Level 2)", 3: "Serious (Level 3)"}
    sev_str = severity_map.get(session.get("severity"), "N/A")
    history = session.get("history") or "None reported"
    symptoms = ", ".join(session.get("symptoms", []))

    patient_data = [
        [
            Paragraph("<b>Age:</b>", body_style), Paragraph(str(age) + " yrs", body_style),
            Paragraph("<b>Gender:</b>", body_style), Paragraph(str(gender), body_style),
            Paragraph("<b>Severity:</b>", body_style), Paragraph(str(sev_str), body_style)
        ],
        [
            Paragraph("<b>Symptoms:</b>", body_style), Paragraph(symptoms, bold_body_style),
            Paragraph("<b>History:</b>", body_style), Paragraph(history, body_style),
            "", ""
        ]
    ]
    patient_table = Table(patient_data, colWidths=[60, 110, 50, 100, 60, 160])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('SPAN', (1, 1), (1, 1)),
        ('SPAN', (3, 1), (5, 1)),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 10))

    # 3. AI Diagnostic Evaluation Section
    story.append(Paragraph("2. AI DIAGNOSTIC EVALUATION (BioBERT / ClinicalBERT Architecture)", section_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=accent_color, spaceBefore=2, spaceAfter=6))

    diseases = recommendation.get("predicted_diseases", [])
    medicines = recommendation.get("recommended_medicines", [])
    primary_disease = diseases[0]["disease"] if diseases else "Undetermined"
    dis_info = recommender_details.get(primary_disease, {})

    dis_table_data = [[
        Paragraph("<b>Condition / Diagnosis</b>", table_header_style),
        Paragraph("<b>AI Confidence Score</b>", table_header_style)
    ]]
    for d in diseases[:3]:
        pct = d["confidence"]
        dis_table_data.append([
            Paragraph(f"<b>{d['disease']}</b>", table_cell_bold),
            Paragraph(f"<b>{pct:.1f}%</b>", table_cell_style)
        ])

    dis_table = Table(dis_table_data, colWidths=[340, 200])
    dis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(dis_table)
    story.append(Spacer(1, 4))

    desc_text = dis_info.get("description", "N/A")
    story.append(Paragraph(f"<b>Clinical Overview:</b> {desc_text}", body_style))
    story.append(Spacer(1, 10))

    # 4. Recommended Medicines & Pharmacological Profile
    story.append(Paragraph(f"3. RECOMMENDED PRESCRIPTION FOR {primary_disease.upper()}", section_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=accent_color, spaceBefore=2, spaceAfter=6))

    med_table_data = [[
        Paragraph("#", table_header_style),
        Paragraph("Medicine", table_header_style),
        Paragraph("Generic Name", table_header_style),
        Paragraph("Drug Class", table_header_style),
        Paragraph("Dosage & Frequency", table_header_style),
        Paragraph("Cat.", table_header_style)
    ]]

    for i, med in enumerate(medicines, 1):
        mname = med.get("medicine", "N/A")
        info = med.get("info", {})
        warn = med.get("warning")

        row_med_name = f"<b>{mname}</b>"
        if warn:
            row_med_name += "<br/><font color='#dc2626'><b>[⚠️ Safety Alert]</b></font>"

        med_table_data.append([
            Paragraph(str(i), table_cell_style),
            Paragraph(row_med_name, table_cell_style),
            Paragraph(info.get("generic_name", "N/A"), table_cell_style),
            Paragraph(info.get("drug_class", "N/A"), table_cell_style),
            Paragraph(info.get("dosage", "N/A"), table_cell_style),
            Paragraph(info.get("category", "N/A"), table_cell_style)
        ])

    med_table = Table(med_table_data, colWidths=[20, 110, 110, 110, 140, 50])
    med_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(med_table)
    story.append(Spacer(1, 6))

    # Any safety warnings box
    warnings_found = [m for m in medicines if m.get("warning")]
    if warnings_found:
        for m in warnings_found:
            warn_text = f"<b>⚠️ Contraindication Alert ({m['medicine']}):</b> Patient history matches contraindication: <i>'{m['warning']}'</i>."
            warn_table = Table([[Paragraph(warn_text, ParagraphStyle("Warn", parent=body_style, textColor=alert_red))]], colWidths=[540])
            warn_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), alert_bg),
                ('BOX', (0, 0), (-1, -1), 1, alert_red),
                ('PADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(warn_table)
            story.append(Spacer(1, 4))

    # 5. Lifestyle Guidance: Precautions, Diet, Workout
    story.append(Spacer(1, 4))
    story.append(Paragraph("4. CLINICAL PRECAUTIONS & LIFESTYLE PLAN", section_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=accent_color, spaceBefore=2, spaceAfter=6))

    precautions_list = dis_info.get("precautions", [])
    diet_list = dis_info.get("diet", [])
    workout_list = dis_info.get("workout", [])

    prec_text = "<br/>".join([f"• {p}" for p in precautions_list]) or "• General rest and monitoring"
    diet_text = "<br/>".join([f"• {d}" for d in diet_list]) or "• Balanced nutritious diet"
    work_text = "<br/>".join([f"• {w}" for w in workout_list]) or "• Adequate bed rest"

    lifestyle_data = [
        [Paragraph("<b>🛡️ Precautions</b>", table_cell_bold), Paragraph("<b>🥗 Dietary Advice</b>", table_cell_bold), Paragraph("<b>🏃 Physical Activity</b>", table_cell_bold)],
        [Paragraph(prec_text, table_cell_style), Paragraph(diet_text, table_cell_style), Paragraph(work_text, table_cell_style)]
    ]

    lifestyle_table = Table(lifestyle_data, colWidths=[180, 180, 180])
    lifestyle_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(lifestyle_table)
    story.append(Spacer(1, 10))

    # 6. Verification Badge & Disclaimer
    footer_data = [
        [
            Paragraph("<b>Verified by:</b> MediRec AI Clinical Engine (HuggingFace Transformers)", body_style),
            Paragraph("<b>Authorized Signature:</b><br/><i>M. Ali Sanwal (FYP Student)</i>", body_style)
        ],
        [
            Paragraph("<font color='#64748b'><b>DISCLAIMER:</b> Non-editable official consultation document generated for CS619 FYP evaluation. Always consult a licensed medical doctor before taking medication.</font>", ParagraphStyle("Disc", parent=body_style, fontSize=7, leading=9)),
            ""
        ]
    ]
    footer_table = Table(footer_data, colWidths=[360, 180])
    footer_table.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), 1, primary_color),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('SPAN', (0, 1), (1, 1)),
    ]))
    story.append(KeepTogether(footer_table))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
