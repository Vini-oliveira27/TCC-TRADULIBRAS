// ==============================================================
// TCC LIBRAS - TRADULIBRAS
// Codigo Arduino para controle da mao robotica em Libras
// Suporte a palavras completas
// ==============================================================

#include <Servo.h>

// ==================== DECLARACAO DOS SERVOS ====================
Servo indicador;
Servo medio;
Servo anelar;
Servo minimo;
Servo polegar;
Servo dedao;
Servo punho;
Servo pulso;

// ==================== DEFINICAO DOS PINOS ====================
const int p_indicador = 2;
const int p_medio = 3;
const int p_anelar = 4;
const int p_minimo = 5;
const int p_polegar = 6;
const int p_dedao = 7;
const int p_punho = 8;
const int p_pulso = 9;

// ==================== CONFIGURACAO DE TIMING ====================
const int delayZ = 500; // Delay aumentado para movimentos sequenciais da letra Z
const int delayEntreLetras = 500; // Delay de 0.5 segundos entre cada letra

// ==================== TABELA DE COORDENADAS ====================
const int coordenadas[26][8] = {
  // A
  {180, 180, 160, 180, 140, 180, 90, 97},
  // B
  {0, 0, 0, 0, 160, 90, 90, 97},
  // C
  {105, 125, 90, 100, 100, 90, 90, 0},
  // D
  {0, 135, 100, 105, 110, 70, 90, 0},
  // E
  {55, 63, 40, 50, 160, 90, 90, 97},
  // F
  {110, 0, 0, 0, 120, 100, 90, 97},
  // G
  {0, 180, 160, 180, 140, 180, 90, 97},
  // H
  {0, 70, 160, 180, 160, 90, 90, 35},
  // I
  {180, 180, 160, 0, 140, 110, 90, 97},
  // J
  {180, 180, 160, 0, 120, 100, 130, 20},
  // K
  {0, 70, 160, 180, 160, 90, 130, 97},
  // L
  {0, 180, 160, 180, 0, 180, 90, 97},
  // M
  {0, 0, 0, 180, 135, 135, 180, 97},
  // N
  {0, 0, 160, 180, 135, 135, 180, 97},
  // O
  {115, 135, 100, 105, 110, 70, 90, 0},
  // P
  {0, 70, 160, 180, 140, 90, 160, 10},
  // Q
  {0, 180, 160, 180, 135, 135, 180, 97},
  // R
  {35, 0, 160, 180, 135, 130, 90, 97},
  // S
  {180, 180, 160, 180, 70, 80, 90, 97},
  // T
  {110, 0, 0, 0, 130, 70, 90, 97},
  // U
  {0, 0, 160, 180, 115, 180, 90, 97},
  // V
  {0, 65, 160, 180, 115, 180, 90, 97},
  // W
  {0, 0, 0, 180, 135, 135, 90, 97},
  // X
  {50, 180, 160, 180, 135, 135, 160, 97},
  // Y
  {180, 180, 160, 0, 30, 180, 90, 97},
  // Z
  {0, 180, 160, 180, 135, 135, 90, 97}
};

// ==================== COORDENADAS PARA OS 4 PONTOS DA LETRA Z ====================
const int z_ponto1[8] = {50, 180, 160, 180, 135, 135, 110, 140};   // Ponto 1: Posicao inicial
const int z_ponto2[8] = {50, 180, 160, 180, 135, 135, 110, 50};  // Ponto 2: Movimento do pulso
const int z_ponto3[8] = {50, 180, 160, 180, 135, 135, 140, 130};  // Ponto 3: Movimento do polegar e pulso
const int z_ponto4[8] = {50, 180, 160, 180, 135, 135, 140, 80};  // Ponto 4: Posicao final

// ==================== FUNCAO DE REPOUSO ====================
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

// ==================== FUNCAO PARA APLICAR COORDENADAS ====================
void aplicarCoordenadas(const int coord[8]) {
  indicador.write(coord[0]);
  medio.write(coord[1]);
  anelar.write(coord[2]);
  minimo.write(coord[3]);
  polegar.write(coord[4]);
  dedao.write(coord[5]);
  punho.write(coord[6]);
  pulso.write(coord[7]);
}

// ==================== FUNCAO ESPECIAL PARA LETRA Z COM 4 PONTOS ====================
void executarLetraZ() {
  Serial.println("EXECUTANDO: Letra Z - Movimento complexo com 4 pontos");
  
  // Ponto 1: Posicao inicial
  Serial.println("Z - Ponto 1: Posicao inicial");
  aplicarCoordenadas(z_ponto1);
  delay(delayZ);
  
  // Ponto 2: Movimento do pulso
  Serial.println("Z - Ponto 2: Movimento do pulso");
  aplicarCoordenadas(z_ponto2);
  delay(delayZ);
  
  // Ponto 3: Movimento combinado
  Serial.println("Z - Ponto 3: Movimento do polegar e ajuste do pulso");
  aplicarCoordenadas(z_ponto3);
  delay(delayZ);
  
  // Ponto 4: Posicao final
  Serial.println("Z - Ponto 4: Posicao final");
  aplicarCoordenadas(z_ponto4);
  delay(delayZ);
  
  Serial.println("Z - Movimento completo com 4 pontos!");
}

// ==================== FUNCAO PARA EXECUTAR LETRA ====================
void executarLetra(char letra) {
  int indice = letra - 'a';
  
  if (indice >= 0 && indice < 26) {
    
    // Se for a letra Z, usa a funcao especial com 4 pontos
    if (letra == 'z') {
      executarLetraZ();
      return;
    }
    
    // Para outras letras, movimento normal
    int* coord = coordenadas[indice];
    aplicarCoordenadas(coord);
    
    Serial.print("EXECUTANDO: Letra ");
    Serial.println(letra);
  }
}

// ==================== CONFIGURACAO INICIAL ====================
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
  
  Serial.println("TRADULIBRAS - Pronto para receber comandos");
  posicaoRepouso();
}

// ==================== LOOP PRINCIPAL ====================
void loop() {
  if (Serial.available() > 0) {
    char receivedChar = Serial.read();
    
    if (receivedChar == '\n' || receivedChar == '\r') {
      return;
    }
    
    // Delay antes de processar cada letra recebida
    delay(delayEntreLetras);
    
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
        Serial.println("ERRO: Comando nao reconhecido");
        break;
    }
    
    delay(300);
  }
}