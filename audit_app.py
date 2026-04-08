"""
Energy Audit Agent - Streamlit App
Powered by Claude API

Run with: streamlit run audit_app.py
"""

import streamlit as st
import anthropic
import json
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime

# ─── Audit Structure ────────────────────────────────────────────────────────

AUDIT_SECTIONS = [
    {
        "id": "project_info",
        "name": "Project Information",
        "description": "Basic information about the building and audit",
        "fields": ["building_name", "address", "building_type", "year_built",
                   "gross_sq_ft", "number_of_floors", "occupancy_schedule",
                   "primary_contact_name", "audit_date"],
        "completion_hint": "I have the building name, address, type, size, age, occupancy schedule, and contact info."
    },
    {
        "id": "utility_data",
        "name": "Utility & Billing Data",
        "description": "Energy consumption history and utility rate information",
        "fields": ["electric_utility", "gas_utility", "avg_monthly_kwh",
                   "avg_monthly_therms", "peak_demand_kw", "utility_rate_per_kwh",
                   "utility_rate_per_therm", "annual_energy_cost",
                   "demand_charge_structure", "any_existing_renewable_credits"],
        "completion_hint": "I have utility providers, average monthly usage for electricity and gas, demand info, and rate structure."
    },
    {
        "id": "building_envelope",
        "name": "Building Envelope",
        "description": "Walls, roof, insulation, windows, and air sealing",
        "fields": ["wall_construction_type", "wall_insulation_r_value",
                   "roof_type", "roof_insulation_r_value", "ceiling_type",
                   "window_type", "window_glazing", "estimated_window_area_pct",
                   "air_sealing_condition", "notable_envelope_issues"],
        "completion_hint": "I have wall and roof construction types with insulation values, window type and condition, and air sealing assessment."
    },
    {
        "id": "hvac",
        "name": "HVAC Systems",
        "description": "Heating, cooling, ventilation, and controls",
        "fields": ["heating_system_type", "heating_fuel", "heating_system_age",
                   "heating_efficiency_rating", "cooling_system_type",
                   "cooling_system_age", "cooling_efficiency_rating",
                   "ventilation_type", "thermostat_controls",
                   "hvac_maintenance_status", "zoning_description",
                   "duct_condition"],
        "completion_hint": "I have heating and cooling system types, ages, efficiency ratings, ventilation approach, controls, and maintenance status."
    },
    {
        "id": "lighting",
        "name": "Lighting Systems",
        "description": "Fixture types, controls, and daylighting",
        "fields": ["primary_interior_fixture_type", "pct_led",
                   "lighting_controls_type", "occupancy_sensors_present",
                   "daylight_controls_present", "exterior_lighting_type",
                   "exterior_controls", "avg_daily_operating_hours",
                   "areas_of_concern"],
        "completion_hint": "I have fixture types, LED percentage, control systems, operating hours, and exterior lighting info."
    },
    {
        "id": "plug_loads",
        "name": "Plug Loads & Equipment",
        "description": "Office equipment, appliances, motors, and process loads",
        "fields": ["major_equipment_list", "server_room_present",
                   "server_room_cooling", "kitchen_equipment",
                   "vending_machines", "elevator_present",
                   "major_motor_loads", "equipment_age_general",
                   "energy_star_equipment_pct", "after_hours_equipment_left_on"],
        "completion_hint": "I have major equipment types, any server/kitchen/elevator loads, equipment age, and after-hours behavior."
    },
    {
        "id": "renewables",
        "name": "Renewables & Generation",
        "description": "Existing or potential solar, storage, and EV charging",
        "fields": ["existing_solar_kw", "solar_system_age",
                   "battery_storage_present", "battery_capacity_kwh",
                   "ev_charging_stations", "ev_charging_level",
                   "roof_condition_for_solar", "utility_net_metering_available",
                   "interest_in_solar", "interest_in_storage",
                   "interest_in_ev_charging"],
        "completion_hint": "I have existing generation assets, EV charging status, roof condition, utility net metering, and client interest in future renewables."
    },
    {
        "id": "observations",
        "name": "Site Observations & Priorities",
        "description": "General observations, low-hanging fruit, and client priorities",
        "fields": ["top_energy_concerns_from_client", "obvious_quick_wins",
                   "deferred_maintenance_noted", "budget_range_for_improvements",
                   "timeline_for_improvements", "decision_maker_info",
                   "any_planned_renovations", "additional_notes"],
        "completion_hint": "I have client priorities, obvious improvement opportunities, budget range, timeline, and any planned renovations."
    }
]

SYSTEM_PROMPT = """You are an expert energy auditor conducting a structured commercial energy audit.
Your job is to gather specific information from the auditor or building owner through natural conversation.

You are currently collecting information for the "{section_name}" section of the audit.
Section description: {section_description}

The specific data fields you need to collect are:
{fields_list}

Guidelines:
- Ask questions naturally and conversationally, not like a rigid form
- Group related questions together when it makes sense
- Ask follow-up questions if an answer is vague or incomplete
- If the user doesn't know something, note it as "unknown" and move on
- Be encouraging and professional
- When you have gathered sufficient information for all the fields listed above,
  end your response with exactly this text on its own line: [SECTION_COMPLETE]
- Do NOT mark a section complete until you genuinely have useful information for most fields
- You may ask clarifying questions if the user's answer is ambiguous

Current section completion signal: "{completion_hint}"
Only add [SECTION_COMPLETE] when you've reached that level of information."""


# ─── Helper Functions ────────────────────────────────────────────────────────

def get_section_system_prompt(section):
    fields_list = "\n".join(f"- {f.replace('_', ' ').title()}" for f in section["fields"])
    return SYSTEM_PROMPT.format(
        section_name=section["name"],
        section_description=section["description"],
        fields_list=fields_list,
        completion_hint=section["completion_hint"]
    )

def extract_section_data(messages, section, client):
    """Ask Claude to extract structured data from the conversation."""
    fields_list = "\n".join(f"- {f}" for f in section["fields"])
    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )

    extraction_prompt = f"""Based on the following conversation about the "{section['name']}" section of an energy audit,
extract all the data that was collected and return it as a valid JSON object.

Use these exact field names as keys:
{fields_list}

If a value was not discussed or is unknown, use null.
Return ONLY the JSON object, no other text.

Conversation:
{conversation_text}"""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": extraction_prompt}]
    )

    try:
        raw = response.content[0].text.strip()
        # Remove markdown code blocks if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return {"raw_conversation": conversation_text}

def save_audit_data(audit_data, project_info):
    """Save audit data to JSON file."""
    building_name = project_info.get("building_name", "unknown_building")
    safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in building_name)
    safe_name = safe_name.replace(' ', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"audit_{safe_name}_{timestamp}.json"

    output = {
        "audit_metadata": {
            "created": datetime.now().isoformat(),
            "app_version": "1.0",
            "sections_completed": list(audit_data.keys())
        },
        "audit_data": audit_data
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    return filename


def send_audit_email(audit_data, filename, project_info):
    """Email the completed audit JSON to chris.slater@gmail.com."""
    gmail_user = st.secrets.get("GMAIL_USER", None)
    gmail_password = st.secrets.get("GMAIL_APP_PASSWORD", None)
    if not gmail_user or not gmail_password:
        return False, "Email credentials not configured in Streamlit secrets."

    recipient = "chris.slater@gmail.com"
    building_name = project_info.get("building_name", "Unknown Building")
    audit_date = datetime.now().strftime("%B %d, %Y")

    # Build a readable plain-text summary for the email body
    summary_lines = [
        f"Energy Audit Completed: {building_name}",
        f"Date: {audit_date}",
        f"Sections completed: {len(audit_data)}",
        "",
        "--- Section Summary ---",
    ]
    for section_id, data in audit_data.items():
        section_name = next((s["name"] for s in AUDIT_SECTIONS if s["id"] == section_id), section_id)
        field_count = sum(1 for v in data.values() if v is not None)
        summary_lines.append(f"  {section_name}: {field_count} fields captured")

    summary_lines += ["", "Full data attached as JSON.", ""]

    msg = MIMEMultipart()
    msg["From"] = gmail_user
    msg["To"] = recipient
    msg["Subject"] = f"Energy Audit — {building_name} — {audit_date}"
    msg.attach(MIMEText("\n".join(summary_lines), "plain"))

    # Attach the JSON file
    json_bytes = json.dumps({"audit_metadata": {"created": datetime.now().isoformat()},
                              "audit_data": audit_data}, indent=2).encode("utf-8")
    attachment = MIMEBase("application", "octet-stream")
    attachment.set_payload(json_bytes)
    encoders.encode_base64(attachment)
    attachment.add_header("Content-Disposition", f'attachment; filename="{filename}"')
    msg.attach(attachment)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, gmail_password)
            server.sendmail(gmail_user, recipient, msg.as_string())
        return True, None
    except Exception as e:
        return False, str(e)


# ─── Streamlit UI ────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Energy Audit Agent",
        page_icon="⚡",
        layout="wide"
    )

    st.title("⚡ Energy Audit Agent")
    st.caption("A structured energy audit assistant powered by Claude AI")

    # ── API Key Setup (checks Streamlit secret first, falls back to manual entry) ──
    if "api_key_set" not in st.session_state:
        st.session_state.api_key_set = False

    if not st.session_state.api_key_set:
        # Try to load key from Streamlit secrets (set via Streamlit Cloud dashboard)
        secret_key = st.secrets.get("ANTHROPIC_API_KEY", None)
        if secret_key and secret_key.startswith("sk-ant-"):
            st.session_state.api_key = secret_key
            st.session_state.api_key_set = True
        else:
            st.info("To get started, enter your Claude API key below. You can get one free at console.anthropic.com")
            api_key = st.text_input("Claude API Key", type="password", placeholder="sk-ant-...")
            if st.button("Start Audit", type="primary"):
                if api_key.startswith("sk-ant-"):
                    st.session_state.api_key = api_key
                    st.session_state.api_key_set = True
                    st.rerun()
                else:
                    st.error("That doesn't look like a valid Claude API key. It should start with 'sk-ant-'")
            return

    # ── Optional Password Gate ───────────────────────────────────────────────
    # If ACCESS_PASSWORD is set in Streamlit secrets, require it before proceeding
    access_password = st.secrets.get("ACCESS_PASSWORD", None)
    if access_password:
        if "access_granted" not in st.session_state:
            st.session_state.access_granted = False
        if not st.session_state.access_granted:
            st.subheader("Access Required")
            entered = st.text_input("Enter access password", type="password")
            if st.button("Continue", type="primary"):
                if entered == access_password:
                    st.session_state.access_granted = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            return

    # ── Session State Init ───────────────────────────────────────────────────
    if "current_section_idx" not in st.session_state:
        st.session_state.current_section_idx = 0
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "audit_data" not in st.session_state:
        st.session_state.audit_data = {}
    if "section_complete" not in st.session_state:
        st.session_state.section_complete = False
    if "audit_complete" not in st.session_state:
        st.session_state.audit_complete = False
    if "saved_file" not in st.session_state:
        st.session_state.saved_file = None

    client = anthropic.Anthropic(api_key=st.session_state.api_key)

    # ── Sidebar Progress ─────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Audit Progress")
        for i, section in enumerate(AUDIT_SECTIONS):
            if i < st.session_state.current_section_idx:
                st.success(f"✓ {section['name']}")
            elif i == st.session_state.current_section_idx and not st.session_state.audit_complete:
                st.info(f"→ {section['name']}")
            else:
                st.text(f"○ {section['name']}")

        if st.session_state.audit_data:
            st.divider()
            st.caption(f"{len(st.session_state.audit_data)}/{len(AUDIT_SECTIONS)} sections complete")

        st.divider()
        if st.button("↩ Restart Audit", type="secondary"):
            for key in list(st.session_state.keys()):
                if key != "api_key" and key != "api_key_set":
                    del st.session_state[key]
            st.rerun()

    # ── Audit Complete Screen ────────────────────────────────────────────────
    if st.session_state.audit_complete:
        st.success("## ✅ Audit Complete!")
        st.write("All sections have been completed.")

        # Send email automatically on first load of this screen
        if "email_sent" not in st.session_state:
            project_info = st.session_state.audit_data.get("project_info", {})
            filename = st.session_state.saved_file or "audit_data.json"
            with st.spinner("Emailing results to chris.slater@gmail.com..."):
                ok, err = send_audit_email(st.session_state.audit_data, filename, project_info)
            if ok:
                st.session_state.email_sent = True
                st.info("📧 Results emailed to chris.slater@gmail.com")
            else:
                st.session_state.email_sent = False
                st.warning(f"Email could not be sent: {err}. Use the download button below.")
        elif st.session_state.email_sent:
            st.info("📧 Results emailed to chris.slater@gmail.com")

        st.subheader("Audit Summary")
        for section_id, data in st.session_state.audit_data.items():
            section_name = next((s["name"] for s in AUDIT_SECTIONS if s["id"] == section_id), section_id)
            with st.expander(f"📋 {section_name}"):
                st.json(data)

        col1, col2 = st.columns(2)
        with col1:
            output = json.dumps(st.session_state.audit_data, indent=2)
            st.download_button(
                label="Download JSON",
                data=output,
                file_name=st.session_state.saved_file or "audit_data.json",
                mime="application/json",
                type="primary"
            )
        with col2:
            if st.button("Start New Audit"):
                for key in list(st.session_state.keys()):
                    if key != "api_key" and key != "api_key_set":
                        del st.session_state[key]
                st.rerun()
        return

    # ── Current Section ──────────────────────────────────────────────────────
    current_section = AUDIT_SECTIONS[st.session_state.current_section_idx]

    st.subheader(f"Section {st.session_state.current_section_idx + 1} of {len(AUDIT_SECTIONS)}: {current_section['name']}")
    st.caption(current_section["description"])

    # Initialize conversation for new section
    if not st.session_state.messages:
        with st.spinner("Starting section..."):
            sys_prompt = get_section_system_prompt(current_section)
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=1000,
                system=sys_prompt,
                messages=[{
                    "role": "user",
                    "content": f"Please begin collecting information for the {current_section['name']} section. Start with your first questions."
                }]
            )
            opener = response.content[0].text.replace("[SECTION_COMPLETE]", "").strip()
            st.session_state.messages = [
                {"role": "user", "content": f"Begin {current_section['name']} section."},
                {"role": "assistant", "content": opener}
            ]
            st.rerun()

    # Display conversation
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user" and "Begin " in msg["content"] and "section." in msg["content"]:
                continue  # Hide the internal trigger message
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # Section complete state
    if st.session_state.section_complete:
        st.success(f"✅ {current_section['name']} section complete!")

        is_last = st.session_state.current_section_idx == len(AUDIT_SECTIONS) - 1
        btn_label = "Complete Audit & Save" if is_last else f"Next: {AUDIT_SECTIONS[st.session_state.current_section_idx + 1]['name']} →"

        if st.button(btn_label, type="primary"):
            # Extract and save section data
            with st.spinner("Saving section data..."):
                section_data = extract_section_data(
                    st.session_state.messages,
                    current_section,
                    client
                )
                st.session_state.audit_data[current_section["id"]] = section_data

            if is_last:
                # Save everything and finish
                project_info = st.session_state.audit_data.get("project_info", {})
                filename = save_audit_data(st.session_state.audit_data, project_info)
                st.session_state.saved_file = filename
                st.session_state.audit_complete = True
                st.rerun()
            else:
                # Move to next section
                st.session_state.current_section_idx += 1
                st.session_state.messages = []
                st.session_state.section_complete = False
                st.rerun()
    else:
        # Chat input
        user_input = st.chat_input("Your response...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.spinner("Thinking..."):
                sys_prompt = get_section_system_prompt(current_section)
                response = client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=1000,
                    system=sys_prompt,
                    messages=st.session_state.messages
                )

                assistant_reply = response.content[0].text

                if "[SECTION_COMPLETE]" in assistant_reply:
                    clean_reply = assistant_reply.replace("[SECTION_COMPLETE]", "").strip()
                    st.session_state.messages.append({"role": "assistant", "content": clean_reply})
                    st.session_state.section_complete = True
                else:
                    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

            st.rerun()


if __name__ == "__main__":
    main()
