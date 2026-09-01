from src.agent import LilyAgent


def main():
    agent = LilyAgent()
    print("🌸 Lily Assistant Ready! (พิมพ์ 'exit' หรือ 'quit' เพื่อออก)")
    print("=" * 55)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nLily: บ๊ายบายค่ะ! แล้วพบกันใหม่นะคะ 🌸✨")
                break

            reply = agent.chat(user_input)
            print(f"\nLily: {reply}")
        except KeyboardInterrupt:
            print("\n\nLily: เซสชันถูกยกเลิก บ๊ายบายค่ะ! 🌸")
            break


if __name__ == "__main__":
    main()
