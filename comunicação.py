# TCC LIBRAS - CONTROLE DE PALAVRAS COMPLETAS
import serial
import time
import serial.tools.list_ports

class MaoRobotica:
    def __init__(self):
        self.arduino = None
        self.porta = None
        
    def conectar(self):
        """Conecta automaticamente com o Arduino"""
        portas = serial.tools.list_ports.comports()
        for porta in portas:
            if 'Arduino' in porta.description or 'CH340' in porta.description:
                self.porta = porta.device
                break
                
        if not self.porta and portas:
            self.porta = portas[0].device
            
        try:
            self.arduino = serial.Serial(self.porta, 115200, timeout=1)
            time.sleep(2)
            print(f"✅ Conectado em {self.porta}")
            return True
        except:
            print(f"❌ Erro ao conectar em {self.porta}")
            return False
    
    def enviar_letra(self, letra):
        """Envia uma letra para o Arduino"""
        if self.arduino and self.arduino.is_open:
            self.arduino.write(letra.encode() + b'\n')
            print(f"📤 Enviado: {letra.upper()}", end=' ')
            
            # Aguarda resposta
            time.sleep(0.8)  # Aumentei o tempo para movimento completo
            resposta = ""
            while self.arduino.in_waiting > 0:
                linha = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                if linha and "EXECUTANDO" in linha:
                    resposta = linha
            
            if resposta:
                print(f"→ {resposta}")
            else:
                print("→ ✓")
                
            return True
        return False
    
    def executar_palavra(self, palavra):
        """Executa uma palavra letra por letra"""
        palavra = palavra.lower().strip()
        
        # Remove caracteres especiais, mantém apenas a-z e espaço
        palavra_limpa = ''.join(c for c in palavra if c.isalpha() or c.isspace())
        
        if not palavra_limpa:
            print("❌ Palavra vazia ou sem letras válidas")
            return False
        
        print(f"\n🎯 EXECUTANDO PALAVRA: '{palavra_limpa.upper()}'")
        print("=" * (len(palavra_limpa) * 3 + 20))
        
        for letra in palavra_limpa:
            if letra == ' ':  # Pausa entre palavras
                print("\n⏸️  ESPAÇO - Aguardando 1 segundo...")
                time.sleep(1)
                continue
                
            if letra.isalpha():
                sucesso = self.enviar_letra(letra)
                if not sucesso:
                    print(f"❌ Erro ao enviar letra {letra}")
                    return False
        
        print("=" * (len(palavra_limpa) * 3 + 20))
        print(f"✅ PALAVRA '{palavra_limpa.upper()}' CONCLUÍDA!")
        return True
    
    def modo_letra_individual(self):
        """Modo para testar letras individualmente"""
        print("\n🔤 MODO LETRA INDIVIDUAL")
        print("Digite letras (a-z) ou '0' para repouso")
        print("Digite 'voltar' para retornar ao menu")
        
        while True:
            comando = input("\n🎯 Digite uma letra: ").strip().lower()
            
            if comando == 'voltar':
                break
            elif comando == '0':
                self.enviar_letra('0')
            elif len(comando) == 1 and comando.isalpha():
                self.enviar_letra(comando)
            else:
                print("❌ Comando inválido! Use a-z ou 0")
    
    def modo_palavras(self):
        """Modo para executar palavras completas"""
        print("\n📝 MODO PALAVRAS COMPLETAS")
        print("Digite palavras ou frases para executar")
        print("Digite 'sair' para retornar ao menu")
        
        while True:
            palavra = input("\n💬 Digite uma palavra/frase: ").strip()
            
            if palavra.lower() == 'sair':
                break
            elif palavra:
                self.executar_palavra(palavra)
                
                # Pergunta se quer continuar
                continuar = input("\n🔄 Executar outra palavra? (s/n): ").strip().lower()
                if continuar != 's':
                    break
            else:
                print("❌ Digite uma palavra válida")
    
    def demonstrar_alfabeto(self):
        """Demonstra todo o alfabeto automaticamente"""
        print("\n🔠 DEMONSTRAÇÃO DO ALFABETO")
        alfabeto = "abcdefghijklmnopqrstuvwxyz"
        
        confirmar = input("Executar demonstração do alfabeto? (s/n): ").strip().lower()
        if confirmar != 's':
            return
        
        print("\n🎬 Iniciando demonstração...")
        print("=" * 50)
        
        for letra in alfabeto:
            self.enviar_letra(letra)
            time.sleep(0.5)  # Pausa entre letras
        
        # Volta para repouso
        self.enviar_letra('0')
        print("=" * 50)
        print("✅ Demonstração concluída!")
    
    def fechar(self):
        """Fecha a conexão"""
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("🔌 Conexão fechada")

# PROGRAMA PRINCIPAL
def main():
    print("🤖 TCC LIBRAS - CONTROLE DE PALAVRAS COMPLETAS")
    print("=" * 50)
    
    robo = MaoRobotica()
    
    if not robo.conectar():
        print("❌ Não foi possível conectar com o Arduino")
        input("Pressione Enter para sair...")
        return
    
    try:
        while True:
            print("\n" + "=" * 50)
            print("🎮 MENU PRINCIPAL")
            print("=" * 50)
            print("1. 🔤 Modo Letra Individual")
            print("2. 📝 Modo Palavras Completas")
            print("3. 🔠 Demonstrar Alfabeto")
            print("4. 🚪 Sair")
            
            opcao = input("\n📋 Escolha uma opção: ").strip()
            
            if opcao == '1':
                robo.modo_letra_individual()
            elif opcao == '2':
                robo.modo_palavras()
            elif opcao == '3':
                robo.demonstrar_alfabeto()
            elif opcao == '4':
                print("👋 Encerrando programa...")
                break
            else:
                print("❌ Opção inválida! Escolha 1-4")
                
    except KeyboardInterrupt:
        print("\n\n⏹️ Programa interrompido pelo usuário")
    finally:
        robo.fechar()

if __name__ == "__main__":
    main()