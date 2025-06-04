#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <BLE2902.h>


//---------------------------------BP cuff section--------------------------------------

int ENA = 33;
int in1 = 25; //1 is positive
int in2 = 26;

int ENB = 13;
int in3 = 14;
int in4 = 12; //4 is positive

int times = 0;
float myTime = 0;
float initialTime;
float reading_complete = false;

#include <SPI.h>
#define SPEED 200000 //clock speed is 200kHz

float press_counts = 0;
float pressure = 0;
float pressmmHg=0;
float pressure_sent;
float outputmax = 15099494;
float outputmin = 1677722;
float pmax = 150;
float pmin = 0;
float initial_pressure=0;
float offset=0;
float sum=0;
float data_points=0;

void BPMeasurements();
void PressureSensorREADING();
void sendPressure(float value);
void sendDoneSignal();

//---------------------------BP cuff section end-------------------------------------------

BLECharacteristic* pDataChar;
BLECharacteristic* pReadyChar;

#define SERVICE_UUID        "12345678-1234-5678-1234-56789abcdef0"
#define WRITE_CHAR_UUID      "abcd0001-1234-5678-1234-56789abcdef0"
#define READY_CHAR_UUID     "abcd0002-1234-5678-1234-56789abcdef0"

BLEServer* pServer = NULL;   

// Some variables to keep track on device connected
bool deviceConnected = false;
bool oldDeviceConnected = false;
bool doneSent = false;
bool startMeasurement = false;
float sys_threshold;
float dias_threshold;
int sys_overlap = sys_threshold + 15;

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
    };

    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
    }
};

class MyCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pCharacteristic) {
    uint8_t* value = pCharacteristic->getData();
    sys_threshold = value[0] + (value[1] << 8);
    dias_threshold = value[2] + (value[3] << 8);
    doneSent = false;
    startMeasurement = true;

    Serial.print("Systolic Threshold: ");
    Serial.println(sys_threshold);
    Serial.print("Diastolic threshold: ");
    Serial.println(dias_threshold);
    
  }
};

void setup() {
  Serial.begin(9600);
  analogReadResolution(10);

  pinMode(ENA, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);

  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);

  pinMode(SS, OUTPUT);
  digitalWrite(SS, HIGH);

  SPI.begin();

  BLEDevice::init("ESP32_POT");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);

  pDataChar = pService->createCharacteristic(
                WRITE_CHAR_UUID,
                BLECharacteristic::PROPERTY_WRITE);
  pDataChar->setCallbacks(new MyCallbacks());
  pService->start();

  pReadyChar = pService->createCharacteristic(
                READY_CHAR_UUID,
                BLECharacteristic::PROPERTY_NOTIFY);
  pReadyChar->addDescriptor(new BLE2902());

  pService->start();
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  BLEDevice::startAdvertising();
  Serial.println("Waiting for central...");

}

void BPMeasurements() {
  delay(5000);
  //calibrating pressure sensor
  for (times=0; times<10; times++) { //collecting 10 points for offset
    
    PressureSensorREADING();
    initial_pressure= pressure * 51.7149;
    delay (1000);
    Serial.print("Pressure not counted: ");
    Serial.print(initial_pressure);
    Serial.println(" mmHg");
  }
  for (times=0; times<10; times++) { //collecting 10 points for offset
    PressureSensorREADING();
    initial_pressure= pressure * 51.7149;
    sum=sum+initial_pressure;
    data_points=data_points+1;
    delay (1000);
    Serial.print("Pressure: ");
    Serial.print(initial_pressure);
    Serial.println(" mmHg");
  }
  offset=sum/data_points;
  Serial.print ("Initial Pressure Readings: ");
  Serial.print (offset);
  Serial.println(" mmHg");
  initialTime=millis();
  
  //Begin Inflating of pump
  while (pressmmHg > -10 && pressmmHg < sys_threshold && deviceConnected) {
  
    //Begin pressure readings
    PressureSensorREADING();
    pressmmHg = (pressure * 51.7149) - offset;
    
    myTime=(millis()-initialTime)/1000;

    // AIR PUMP
    analogWrite(ENA, 255); 
    analogWrite(ENB, 255);

    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW); 

    Serial.print(myTime);
    Serial.print (",");
    Serial.print ("CASE 1");
    Serial.print (",");
    Serial.println(pressmmHg);
    delay(10);
  }
//--
//Begin Slow Deflation
  while (pressmmHg < sys_threshold+15 && pressmmHg > dias_threshold && deviceConnected) { 
    analogWrite(ENA, 0);
    
    PressureSensorREADING();
    pressmmHg = (pressure * 51.7149) - offset;
    pressure_sent = pressmmHg;
    Serial.println("Data collection started");
 
    myTime=(millis()-initialTime)/1000;
    sendPressure(pressure_sent);
    Serial.print(sys_overlap);
    Serial.println(myTime);
    Serial.print(",");
    Serial.print ("CASE 2");
    Serial.print (",");
    Serial.println(pressmmHg);
    delay (10);

  }

//Complete Deflation
  while (pressmmHg>=0 && deviceConnected) {
    analogWrite (ENB, 0);
    
    PressureSensorREADING();
    pressmmHg = (pressure * 51.7149) - offset;
 
    myTime=(millis()-initialTime)/1000;
    Serial.print(myTime);
    Serial.print(",");
    Serial.print ("CASE 3");
    Serial.print (",");
    Serial.println(pressmmHg);
    delay (10);
    
  }
  sendDoneSignal();
  reading_complete = true;
  Serial.println("Reading Sent to GUI");
//---------
}

  
//---------------------END of BP cuff section-----------------------------------------------------------------------------------------------------------------------------------------

void loop() {
  if (deviceConnected && startMeasurement && !doneSent) {
    BPMeasurements();
    startMeasurement = false;
    oldDeviceConnected = deviceConnected;
  }

  if (!deviceConnected && oldDeviceConnected) {
    delay(500); // give the bluetooth stack the chance to get things ready
    pServer->startAdvertising(); // restart advertising
    Serial.println("start advertising");
    doneSent = false;
    oldDeviceConnected = deviceConnected;
  }
  if (!deviceConnected && !oldDeviceConnected) {
    delay(500); // give the bluetooth stack the chance to get things ready
    pServer->startAdvertising(); // restart advertising
    Serial.println("start advertising");
    doneSent = false;
  }
  // Connecting
  if (deviceConnected && !oldDeviceConnected) {
    // do stuff here on connecting
    oldDeviceConnected = deviceConnected;
    doneSent = false;
  }
}


void sendPressure(float value) {
  uint8_t scaled = (uint8_t)(value * 10);
  uint8_t bytes[2];
  bytes[0] = scaled & 0xFF;
  bytes[1] = (scaled >> 8) & 0xFF;
  pReadyChar->setValue(bytes, 2);
  pReadyChar->notify();
  Serial.printf("Sent pressure: %.1f mmHg\n", value);
}

void sendDoneSignal() {
  uint8_t doneSignal[2] = {0xFF, 0xFF};
  pReadyChar->setValue(doneSignal, 2);
  pReadyChar->notify();
  Serial.println("Measurement complete, sent stop signal.");
  doneSent = true;
}


  //---------------------------BP_cuff section----------------------------------------------------------------------------------------------------

void PressureSensorREADING() {
    uint8_t data[4] = {0xF0, 0x00, 0x00, 0x00};
    uint8_t cmd[3] = {0xAA, 0x00, 0x00};

    SPI.beginTransaction(SPISettings(SPEED, MSBFIRST, SPI_MODE0));
    digitalWrite(SS,LOW);
    SPI.transfer(cmd, 3);
    digitalWrite(SS,HIGH);

    digitalWrite(SS, LOW);
    SPI.transfer(data, 4);
    digitalWrite(SS, HIGH);
    SPI.endTransaction ();

    press_counts = (double)((int32_t)data[3] + (int32_t)data[2]* (int32_t)256 + (int32_t)data[1]*(int32_t)65536);
    pressure = (((press_counts - outputmin) * (pmax - pmin)) / (outputmax - outputmin))+ pmin;
    //-------------------------------------END of BP cuff section-------------------------------------------
}
