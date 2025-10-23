// ==============================================================
// TCC LIBRAS - TRADULIBRAS
// Código Arduino para controle da mão robótica em Libras
// Suporte a palavras completas
// ==============================================================

#include <Servo.h>

// ==================== DECLARAÇÃO DOS SERVOS ====================
Servo indicador;
Servo medio;
Servo anelar;
Servo minimo;
Servo polegar;
Servo dedao;
Servo punho;
Servo pulso;

// ==================== DEFINIÇÃO DOS PINOS ====================
const int p_indicador = 2;
const int p_medio = 3;
const int p_anelar = 4;
const int p_minimo = 5;
const int p_polegar = 6;
const int p_dedao = 7;
const int p_punho = 8;
const int p_pulso = 9;

// ==================== CONFIGURAÇÃO DE TIMING ====================
const int delayZ = 300; // Delay para movimentos sequenciais da letra Z

// ==================== TABELA DE COORDENADAS ====================
const int coordenadas[26][8] = {
  {180, 180, 180, 180, 115, 180, 90, 97}, // A
  {0, 0, 0, 0, 140, 90, 90, 97},          // B
  {130, 125, 115, 100, 80, 90, 90, 0},    // C
  {0, 145, 140, 125, 100, 70, 90, 0},     // D
  {80, 75, 60, 50, 140, 90, 90, 97},      // E
  {130, 0, 0, 0, 120, 100, 90, 97},       // F
  {0, 180, 180, 180, 135, 135, 90, 97},   // G
  {0, 70, 180, 180, 140, 90, 90, 35},     // H
  {180, 180, 180, 0, 120, 100, 90, 97},   // I
  {180, 180, 180, 0, 120, 100, 130, 20},  // J
  {0, 70, 180, 180, 140, 90, 130, 97},    // K
  {0, 180, 180, 180, 0, 180, 90, 97},     // L
  {0, 0, 0, 180, 135, 135, 180, 97},      // M
  {0, 0, 180, 180, 135, 135, 180, 97},    // N
  {145, 145, 140, 125, 100, 70, 90, 0},   // O
  {0, 70, 180, 180, 140, 90, 160, 10},    // P
  {0, 180, 180, 180, 135, 135, 180, 97},  // Q
  {0, 65, 180, 180, 140, 90, 90, 97},     // R
  {180, 180, 180, 180, 90, 80, 90, 97},   // S
  {135, 0, 0, 0, 120, 65, 90, 97},        // T
  {0, 0, 180, 180, 115, 180, 90, 97},     // U
  {0, 100, 180, 180, 140, 90, 90, 97},    // V
  {0, 0, 0, 180, 135, 135, 90, 97},       // W
  {85, 180, 180, 180, 135, 135, 160, 97}, // X
  {180, 180, 180, 0, 30, 180, 90, 97},    // Y
  {0, 180, 180, 180, 135, 135, 110, 45}    // Z - Movimento padrão (será sobrescrito pela função especial)
};

// ==================== FUNÇÃO DE REPOUSO ====================
void posicaoRepouso() {
  indicador.write(0); 
  medio.write(0);
  anelar.write(0);
  minimo.write(0);
  polegar.write(0);
  dedao.write(90);
  punho.write(90);
  pulso.write(90);
  Serial.println("REPOUSO: Mao em posicao de repouso");
}

// ==================== FUNÇÃO ESPECIAL PARA LETRA Z ====================
void executarLetraZ() {
  Serial.println("EXECUTANDO: Letra Z - Movimento complexo");
  
  // Primeira etapa do movimento da letra Z
  Serial.println("Z - Etapa 1: Preparando dedos");
  indicador.write(0);
  medio.write(180);
  anelar.write(180);
  minimo.write(180);
  delay(delayZ);
  
  // Segunda etapa - movimento do polegar
  Serial.println("Z - Etapa 2: Movimento do polegar");
  polegar.write(135);
  dedao.write(135);
  delay(delayZ);
  
  // Terceira etapa - movimento do pulso/punho (se necessário)
  Serial.println("Z - Etapa 3: Ajuste final");
  punho.write(90);
  pulso.write(97);
  delay(delayZ * 2);
  
  Serial.println("Z - Movimento completo!");
}

// ==================== FUNÇÃO PARA EXECUTAR LETRA ====================
void executarLetra(char letra) {
  int indice = letra - 'a';
  
  if (indice >= 0 && indice < 26) {
    
    // Se for a letra Z, usa a função especial
    if (letra == 'z') {
      executarLetraZ();
      return;
    }
    
    // Para outras letras, movimento normal
    int* coord = coordenadas[indice];
    
    indicador.write(coord[0]);
    medio.write(coord[1]);
    anelar.write(coord[2]);
    minimo.write(coord[3]);
    polegar.write(coord[4]);
    dedao.write(coord[5]);
    punho.write(coord[6]);
    pulso.write(coord[7]);
    
    Serial.print("EXECUTANDO: Letra ");
    Serial.println(letra);
  }
}

// ==================== CONFIGURAÇÃO INICIAL ====================
void setup() {
  indicador.attach(p_indicador);
  medio.attach(p_medio);
  anelar.attach(p_anelar);
  minimo.attach(p_minimo);
  polegar.attach(p_polegar);
  dedao.attach(p_dedao);
  punho.attach(p_punho);
  pulso.attach(p_pulso);
  
  Serial.begin(115200);
  while (!Serial) {
    ; // Aguarda porta serial
  }
  
  Serial.println("INICIO: TRADULIBRAS - MAO ROBOTICA INICIADA");
  Serial.println("MODOS: Letras individuais e palavras completas");
  Serial.println("AGUARDANDO: Comandos...");
  
  posicaoRepouso();
}

// ==================== LOOP PRINCIPAL ====================
void loop() {
  if (Serial.available() > 0) {
    char receivedChar = Serial.read();
    
    if (receivedChar == '\n' || receivedChar == '\r') {
      return;
    }
    
    Serial.print("RECEBIDO: ");
    Serial.println(receivedChar);
    
    switch (receivedChar) {
      case 'a': case 'b': case 'c': case 'd': case 'e': case 'f': case 'g':
      case 'h': case 'i': case 'j': case 'k': case 'l': case 'm': case 'n':
      case 'o': case 'p': case 'q': case 'r': case 's': case 't': case 'u':
      case 'v': case 'w': case 'x': case 'y': case 'z':
        executarLetra(receivedChar);
        break;
      
      case '0':
        posicaoRepouso();
        break;
      
      default:
        Serial.println("ERRO: Comando não reconhecido");
        break;
    }
    
    delay(300);
  }
}