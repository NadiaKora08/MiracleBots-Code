#include <BLEDevice.h>
22212018191716151413121110987654321
BLEServer* pServer = NULL;


// Some variables to keep track on device connected
bool deviceConnected = false;
bool oldDeviceConnected = false;
bool administrationDone = false;
double rpm;
float duration;
float initialTime;

Not connected. Select a board and a port to connect automatically.
New Line

#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
//-------------------------MODBUS-----------------------------------------------------------
#include <ModbusMaster.h>
//coil register MAKE SURE TO USE writeSingleCoil(address,value)
#define ONOFF 0x0001 //start or stop
#define Motor_Direction 0x0002 //changing motor direction
// holding register (write only)
#define BAUDRATE 0x4002
#define SUBD 0x4003 //subdivision
#define SPEED1 0x4009 //target speed
#define SPEED2 0x400A

//input register (read only)
#define ACTSPEED1 0x3001 //output actual speed
#define ACTSPEED2 0x3002


//-----------------------MODBUS END-------------------------------------------------------------

//-----------------------SERVO MOTOR------------------------------------------------------------
#include <ESP32Servo.h>
Servo myservo;
int pos =0;
int servoPin =32;
int medication =1;
////myservo.setPeriodHertz(50);
//ESP32PWM::allocateTimer(0);
//ESP32PWM::allocateTimer(1);
//ESP32PWM::allocateTimer(2);
//ESP32PWM::allocateTimer(3);
//myservo.setPeriodHertz(50);

//-----------------------SERVO END-------------------------------------------------------------
#define SERVICE_UUID      "87654321-4321-6789-4321-fedcba987654"
#define WRITE_CHAR_UUID   "abcd1001-4321-6789-4321-fedcba987654"
#define LED_PIN 2

BLEServer* pServer = NULL;


// Some variables to keep track on device connected
bool deviceConnected = false;
bool oldDeviceConnected = false;
bool administrationDone = false;
double rpm;
float duration;
float initialTime;
float myTime = 0;
bool MotorOnTwo = false;
volatile bool interruptRequested = false;
bool administrationFullyDone = false;
unsigned long lastWriteTime = 0;
const unsigned long debounceInterval = 60000;
unsigned long now = millis();

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

    myTime = 0;
    initialTime = millis();

    uint16_t rpm_raw = value[0] + (value[1] << 8);
    rpm = rpm_raw / 1000.0;

    uint16_t duration_raw = value[2] + (value[3] << 8);
    duration = duration_raw*60;

    medication = value[4];
    
    Serial.print("Received rpms: ");
    Serial.println(rpm);
    Serial.print("Received duration:");
    Serial.println(duration);
    Serial.print("Received medication type:");
    Serial.println(medication);
    Serial.print(administrationDone);

    if (medication == 2 && !MotorOnTwo) {
      for (pos=0;pos<=105;pos+=1){
        myservo.write(pos);
        delay (15);
        MotorOnTwo = true;
        Serial.println("Motor Turned");
      }
      delay(1000);
    }
    if (medication == 1 && MotorOnTwo) {
      for (pos=105;pos>=0;pos-=1){
        myservo.write(pos);
        delay (15);
        MotorOnTwo = false;
        Serial.println("Motor Turned Back");
      }
    delay(1000);
    }
    Serial.print("now:");
    Serial.println(now);
    Serial.print("lastWriteTime:");
    Serial.println(lastWriteTime);
    Serial.print("oldDeviceConnected:");
    Serial.println(oldDeviceConnected);
    Serial.print("administrationDone");
    Serial.println(administrationDone);
    if ((now - lastWriteTime > debounceInterval) && oldDeviceConnected && !administrationDone) {
        interruptRequested = true;
        Serial.println("Interrupt marked true (debounced)");
    } 
    else {
        Serial.println("Ignored write due to cooldown or already done");
    }
    lastWriteTime = now;
    oldDeviceConnected = true;
    Serial.print("Old Device marked true");
  }
};


//----------------------------------------------MODBUS-------------------------------------------------------------------

ModbusMaster node;
int bp = 1;
uint32_t speed;
//uint32_t actualspeed;
float actualspeed;
const int enTX=18;

void preTransmission (){
  digitalWrite(enTX,HIGH);
}

void postTransmission(){
  digitalWrite(enTX,LOW);
}

//-------------------------------------------------MODBUS END------------------------------------------------------------------

void setup() {
  Serial.begin(115200);
//--------------------------------------------------MODBUS------------------------------------------------
  Serial1.begin(9600);
  Serial2.begin(9600,SERIAL_8N1, 16,17);
  node.begin(1,Serial2);
  node.preTransmission(preTransmission);
  node.postTransmission(postTransmission);
  node.writeSingleRegister(BAUDRATE,3);
  pinMode(enTX, OUTPUT);
  digitalWrite (enTX,HIGH);
  pinMode(LED_PIN, OUTPUT);

//------------------------------------------------MODBUS END------------------------------------------------

  BLEDevice::init("ESP32_LED");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());
  BLEService *pService = pServer->createService(SERVICE_UUID);

  BLECharacteristic *pChar = pService->createCharacteristic(
                              WRITE_CHAR_UUID,
                              BLECharacteristic::PROPERTY_WRITE);
  pChar->setCallbacks(new MyCallbacks());
  pService->start();

  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  BLEDevice::startAdvertising();
  Serial.println("Ready to receive average value...");

  //-----------------------------------------SERVO ----------------------------------------------------
  myservo.setPeriodHertz(50);
  myservo.attach(servoPin);
  // ----------------------------------SERVO END--------------------------------------------------------
}

void loop() {
  while (myTime < (duration * 1000 - 110000) && !interruptRequested) {
    Serial.println(myTime);
    node.writeSingleCoil(ONOFF,0); //prevent from running default 350rpm
    node.writeSingleCoil(Motor_Direction,1);
    
    speed=100*rpm;
    node.setTransmitBuffer(1, lowWord(speed)); //1st index goes in second, lowWord is 0x5678
    node.setTransmitBuffer(0, highWord(speed)); // 0 index goes in first, highWord is 0x1234
    node.writeMultipleRegisters(SPEED1,2);
    node.writeSingleCoil(ONOFF,1);
    Serial.print("Pump running");
    for (int i = 0; i < 60000 && !interruptRequested; i += 1000) {
        delay(1000);
    }

    node.writeSingleCoil(ONOFF, 0);

    Serial.print("Pump stopped");

    for (int i = 0; i < 60000 && !interruptRequested; i += 1000) {
        delay(1000);
    }

    Serial.println("Pump finished running, entering new loop");
    administrationDone = true;
    myTime=millis()-initialTime;
  }

  if (administrationDone && MotorOnTwo) {
      for (pos=105;pos>=0;pos-=1){
        myservo.write(pos);
        delay (15);
        Serial.println(myTime);
        Serial.println("Turning Back");
      }
      MotorOnTwo = false;
      administrationDone = false;
      administrationFullyDone = true;
      lastWriteTime = now;
      Serial.println(myTime);
      Serial.print("Motor Turned Back");
  }

  if (interruptRequested) {
    Serial.println("Loop interrupted by new BLE values!");
    node.writeSingleCoil(ONOFF, 0); // Stop pump
    interruptRequested = false;
    return;  // Exit to top of loop() cleanly
  }

  if (!deviceConnected && oldDeviceConnected) {
    delay(500); // give the bluetooth stack the chance to get things ready
    pServer->startAdvertising(); // restart advertising
    Serial.println("start advertising");
    oldDeviceConnected = false;
  }
  if (!deviceConnected && !oldDeviceConnected) {
    delay(500); // give the bluetooth stack the chance to get things ready
    pServer->startAdvertising(); // restart advertising
    Serial.println("start advertising");
  }
    // Connecting
  if (deviceConnected && !oldDeviceConnected) {
    // do stuff here on connecting
    oldDeviceConnected = deviceConnected;
    Serial.println("Device Reconnected, Going Back");
    return;
  }
}

//3: Did not figure out how to interrupt the BLEn in the middle of it. The main problem is value set true (figure out how to only turn is true on the second run)
