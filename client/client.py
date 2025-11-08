import sys, json, requests

def main():
    if len(sys.argv) < 3:
        print("Uso: python client.py http://HOST:PORT 'Song A' 'Song B' ...")
        sys.exit(1)
    url = sys.argv[1].rstrip("/") + "/api/recommend"
    songs = sys.argv[2:]
    r = requests.post(url, json={"songs": songs}, timeout=60)
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
