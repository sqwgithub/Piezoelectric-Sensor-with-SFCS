#include "Arduino.h"
#define SET(x,y) (x |=(1<<y))        //-Bit set/clear macros
#define CLR(x,y) (x &= (~(1<<y)))         // |
#define CHK(x,y) (x & (1<<y))             // |
#define TOG(x,y) (x^=(1<<y))              //-+
int FREQNUM=200;
unsigned long startTime = millis();
int sweep(int freq){
    CLR(TCCR1B, 0);
    TCNT1 = 0;
    ICR1 = freq;
    OCR1A = freq/2;
    SET(TCCR1B, 0);
    delayMicroseconds(1);
}
void setup() {
    Serial.begin(115200);
    TCCR1A=0b10000010;        //-Set up frequency generator
    TCCR1B=0b00011001;        //-+
    ICR1=110;
    OCR1A=55;
    pinMode(11, OUTPUT);
    pinMode(12, OUTPUT);
    pinMode(A0, INPUT);
    pinMode(A9, INPUT);
    pinMode(A10, INPUT);
    pinMode(A11, INPUT);
    pinMode(A12, INPUT);
    
    
}

void loop() {
  // 执行频率扫描
  static float readings[400];
  for(int i = 3; i < FREQNUM; i++)
  {
    
    
    sweep(i);
    readings[i - 3] = analogRead(A0);
//    Serial.print( "s" );
//    Serial.print( (16000 / (i + 1)) );
//    Serial.print( "," );
//    delayMicroseconds(1);
//    Serial.print( analogRead(A0));
//    Serial.print( "," );
//    Serial.print( readings[i - 3]);
//    Serial.print( "," );
//    delayMicroseconds(1);
//    Serial.print( analogRead(A0));;
//    Serial.println( "e" );

  }     
  Serial.print( '#' );
  for (int j = 0; j < FREQNUM-3; j++) {
    
    Serial.print(readings[j]);
    if (j < FREQNUM-4){
      Serial.print( ',' );
    }
}
Serial.println( '@' );

  Serial.print( '%' );

    Serial.print(analogRead(A9));

      Serial.print( ',' );
    Serial.print(analogRead(A10));

      Serial.print( ',' );
          Serial.print(analogRead(A11));

      Serial.print( ',' );
          Serial.print(analogRead(A12));

Serial.println( '!' );
delay(1);
//while (millis() - startTime < 5000) {
//  startTime = millis();
//  Serial.print( 'f' );
//  Serial.print( FREQNUM );
//  Serial.println( 'f' );
//    // Non-blocking delay to avoid interrupting the process
//}
}
