from road38.play_live import parse_prow
for t in ["NK N. Kanté CDM/CMAge 30 · #7 78 66 75 82 87 83 90",
          "RL R. Lukaku STAge 28 · #9 84 87 74 78 39 83 88",
          "TS Thiago Silva CBAge 36 · #6 53 54 72 72 86 78 85"]:
    print(parse_prow(t))
