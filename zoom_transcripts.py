"""
Grabs zoom transcripts
"""
import re, requests
import sys

def get_zoom_token(account_id, client_id, client_secret):
    import base64
    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    r = requests.post("https://zoom.us/oauth/token",
                      params={"grant_type": "account_credentials", "account_id": account_id},
                      headers={"Authorization": f"Basic {creds}"})
    return r.json()["access_token"]

def download_transcript(meeting_id, token):
    recs = requests.get(
        f"https://api.zoom.us/v2/meetings/{meeting_id}/recordings",
        headers={"Authorization": f"Bearer {token}"}
    ).json()
    for f in recs.get("recording_files", []):
        if f["file_type"] == "TRANSCRIPT":
            vtt = requests.get(f["download_url"] + f"?access_token={token}").text
            return vtt_to_text(vtt)

def vtt_to_text(vtt):
    lines = [l for l in vtt.splitlines() if l and not l.startswith("WEBVTT")
             and not re.match(r'^\d+$', l) and not re.match(r'[\d:]+\s-->', l)]
    return " ".join(lines)

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    meeting_id = sys.argv[1].lower()

    