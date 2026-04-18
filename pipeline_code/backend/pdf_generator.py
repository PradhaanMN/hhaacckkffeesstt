"""
PDF Report Generator for Solar Panel Detection Results
Generates professional PDF reports with statistics and detection results
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    Image, PageBreak, KeepTogether
)
from reportlab.pdfgen import canvas
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Any


def create_pdf_report(
    result_data: Dict[str, Any],
    overlay_path: str,
    output_path: str,
    include_statistics: bool = True
) -> str:
    """
    Generate a professional PDF report for solar panel detection.
    
    Args:
        result_data: Detection result dictionary
        overlay_path: Path to overlay image
        output_path: Output PDF file path
        include_statistics: Include detailed statistics section
    
    Returns:
        Path to generated PDF file
    """
    # Create PDF document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Container for PDF elements
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#7f8c8d'),
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=6
    )
    
    # Header
    elements.append(Paragraph("🌞 Solar Panel Detection Report", title_style))
    elements.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        ParagraphStyle('date', parent=normal_style, alignment=TA_CENTER, fontSize=10, textColor=colors.grey)
    ))
    elements.append(Spacer(1, 0.3*inch))
    
    # Executive Summary Box
    summary_data = [
        ['Detection Status', '✓ SOLAR DETECTED' if result_data['has_solar'] else '✗ NO SOLAR DETECTED'],
        ['Confidence', f"{result_data['confidence']*100:.1f}%"],
        ['Panel Area', f"{result_data['pv_area_sqm_est']:.2f} m²"],
        ['QC Status', result_data['qc_status']],
    ]
    
    summary_table = Table(summary_data, colWidths=[2.5*inch, 3.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
    ]))
    
    elements.append(Paragraph("Executive Summary", heading_style))
    elements.append(summary_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Location Details
    elements.append(Paragraph("Location Details", heading_style))
    
    location_data = [
        ['Sample ID', str(result_data['sample_id'])],
        ['Latitude', f"{result_data['lat']:.6f}°"],
        ['Longitude', f"{result_data['lon']:.6f}°"],
        ['Buffer Zone', f"{result_data['buffer_radius_sqft']} sq.ft"],
        ['Euclidean Distance', f"{result_data.get('euclidean_distance_m_est', 0):.2f} m"],
    ]
    
    location_table = Table(location_data, colWidths=[2.5*inch, 3.5*inch])
    location_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#ecf0f1')),
    ]))
    
    elements.append(location_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Detection Visualization
    if overlay_path and Path(overlay_path).exists():
        elements.append(Paragraph("Detection Visualization", heading_style))
        
        # Add overlay image
        img = Image(overlay_path, width=5.5*inch, height=5.5*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.2*inch))
        
        # Image caption
        caption = Paragraph(
            "<i>Detection overlay showing identified solar panels within buffer zone</i>",
            ParagraphStyle('caption', parent=normal_style, alignment=TA_CENTER, fontSize=9, textColor=colors.grey)
        )
        elements.append(caption)
        elements.append(Spacer(1, 0.3*inch))
    
    # Power Generation Estimates (if solar detected)
    if result_data['has_solar'] and result_data.get('power_estimate'):
        elements.append(Paragraph("Power Generation Estimates", heading_style))
        
        power_est = result_data['power_estimate']
        power_data = [
            ['Peak Capacity', f"{power_est['peak_power_kw']:.2f} kW"],
            ['Daily Energy', f"{power_est['daily_energy_kwh']:.1f} kWh"],
            ['Monthly Energy', f"{power_est['monthly_energy_kwh']:.0f} kWh"],
            ['Yearly Energy', f"{power_est['yearly_energy_kwh']:.0f} kWh"],
        ]
        
        power_table = Table(power_data, colWidths=[2.5*inch, 3.5*inch])
        power_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#e8f5e9')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#c8e6c9')),
        ]))
        
        elements.append(power_table)
        elements.append(Spacer(1, 0.1*inch))
        
        # Assumptions note
        assumptions = Paragraph(
            "<i>Based on 18% panel efficiency, 5.5 peak sun hours/day, 80% system efficiency</i>",
            ParagraphStyle('note', parent=normal_style, fontSize=9, textColor=colors.grey, alignment=TA_LEFT)
        )
        elements.append(assumptions)
        elements.append(Spacer(1, 0.3*inch))
    
    # Technical Statistics
    if include_statistics:
        elements.append(Paragraph("Technical Statistics", heading_style))
        
        stats_data = [
            ['Processing Time', f"{result_data.get('processing_time_seconds', 0):.2f} seconds"],
            ['Image Resolution', f"{result_data.get('image_metadata', {}).get('resolution', 'N/A')}"],
            ['Detection Algorithm', 'YOLOv8 Ensemble (6 models)'],
            ['Buffer Method', 'Dynamic (1200-2400 sq.ft)'],
        ]
        
        stats_table = Table(stats_data, colWidths=[2.5*inch, 3.5*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#ecf0f1')),
        ]))
        
        elements.append(stats_table)
        elements.append(Spacer(1, 0.5*inch))
    
    # JSON Data Section
    elements.append(Paragraph("Raw JSON Data", heading_style))
    
    # Format JSON data nicely
    json_str = json.dumps(result_data, indent=2)
    # Split into lines and limit length
    json_lines = json_str.split('\n')
    
    # Create a code-style paragraph for JSON
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=normal_style,
        fontSize=8,
        fontName='Courier',
        textColor=colors.HexColor('#2c3e50'),
        leftIndent=10,
        rightIndent=10,
        spaceAfter=6,
        leading=10,
        backColor=colors.HexColor('#f8f9fa')
    )
    
    # Add JSON content (limit to reasonable size)
    max_lines = 50
    for i, line in enumerate(json_lines[:max_lines]):
        # Escape special characters for reportlab
        line = line.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
        elements.append(Paragraph(f"<font name='Courier'>{line}</font>", code_style))
    
    if len(json_lines) > max_lines:
        elements.append(Paragraph(
            f"<i>... ({len(json_lines) - max_lines} more lines truncated)</i>",
            ParagraphStyle('truncated', parent=normal_style, fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
        ))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Footer
    footer_text = Paragraph(
        "<b>NeuralStack</b><br/>"
        "PM Surya Ghar: Muft Bijli Yojana - AI-Powered Rooftop PV Detection<br/>"
        "<i>This report is generated automatically using advanced AI detection algorithms.</i>",
        ParagraphStyle(
            'footer',
            parent=normal_style,
            fontSize=9,
            textColor=colors.grey,
            alignment=TA_CENTER,
            borderPadding=10,
            borderColor=colors.HexColor('#bdc3c7'),
            borderWidth=1
        )
    )
    elements.append(footer_text)
    
    # Build PDF
    doc.build(elements)
    
    return output_path


def create_batch_pdf_report(
    results_list: List[Dict[str, Any]],
    output_path: str
) -> str:
    """
    Generate a batch PDF report with statistics for multiple detections.
    
    Args:
        results_list: List of detection result dictionaries
        output_path: Output PDF file path
    
    Returns:
        Path to generated PDF file
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=22,
        alignment=TA_CENTER,
        spaceAfter=30
    )
    elements.append(Paragraph("Batch Detection Report", title_style))
    elements.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y')}",
        ParagraphStyle('date', parent=styles['Normal'], alignment=TA_CENTER, textColor=colors.grey)
    ))
    elements.append(Spacer(1, 0.3*inch))
    
    # Summary Statistics
    total_locations = len(results_list)
    solar_detected = sum(1 for r in results_list if r['has_solar'])
    total_area = sum(r['pv_area_sqm_est'] for r in results_list if r['has_solar'])
    avg_confidence = sum(r['confidence'] for r in results_list) / total_locations if total_locations > 0 else 0
    
    summary_data = [
        ['Total Locations Analyzed', str(total_locations)],
        ['Solar Panels Detected', f"{solar_detected} ({solar_detected/total_locations*100:.1f}%)"],
        ['Total Panel Area', f"{total_area:.2f} m²"],
        ['Average Confidence', f"{avg_confidence*100:.1f}%"],
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#e3f2fd')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.blue),
    ]))
    
    elements.append(Paragraph("Summary Statistics", styles['Heading2']))
    elements.append(summary_table)
    elements.append(Spacer(1, 0.4*inch))
    
    # Detailed Results Table
    elements.append(Paragraph("Detailed Results", styles['Heading2']))
    
    table_data = [['Sample ID', 'Lat', 'Lon', 'Solar', 'Confidence', 'Area (m²)']]
    
    for result in results_list[:50]:  # Limit to first 50 for readability
        table_data.append([
            str(result['sample_id']),
            f"{result['lat']:.4f}",
            f"{result['lon']:.4f}",
            '✓' if result['has_solar'] else '✗',
            f"{result['confidence']*100:.1f}%",
            f"{result['pv_area_sqm_est']:.2f}"
        ])
    
    results_table = Table(table_data, colWidths=[1.2*inch, 0.9*inch, 0.9*inch, 0.7*inch, 0.9*inch, 0.8*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))
    
    elements.append(results_table)
    
    # Build PDF
    doc.build(elements)
    
    return output_path

