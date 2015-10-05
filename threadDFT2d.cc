// Threaded two-dimensional Discrete FFT transform
// Amit Kulkarni
// ECE6122 Project 2

#include <iostream>
#include <string>
#include <math.h>
#include <pthread.h> // Header for pthreads functions

#include "Complex.h"
#include "InputImage.h"

using namespace std;

// Global variables
// Image parameters
int ht, wd;
Complex *d; // Input image data array
Complex *d_inv;
Complex W[512];   // Forward twiddle factor array
Complex W_inv[512]; // Inverse twiddle factor array
unsigned N = 1024;
int imageHeight;
long int imageSize;
double theta;

// Barrier parameters
int nThreads = 16;
int P0; // Total threads for Barrier functions
int count;    // Number of threads in the barrier.
int threadCounter();
pthread_mutex_t countMutex,exitMutex;
pthread_cond_t exitCond;
pthread_mutex_t startCountMutex;
int threadCount;
bool* localSense;
bool globalSense;

//int debugCount = 1; // debug counter for thread count.

// Function to reverse bits in an unsigned integer
// This assumes there is a global variable N that is the
// number of points in the 1D transform.
unsigned ReverseBits(unsigned v)
{ //  Provided to students
  unsigned n = N; // Size of array (which is even 2 power k value)
  unsigned r = 0; // Return value
  //cout<<"Inside RB"<<endl;
  for (--n; n > 0; n >>= 1)
    {
      r <<= 1;        // Shift return value
      r |= (v & 0x1); // Merge in next bit
      v >>= 1;        // Shift reversal value
    }
  return r;
}

// GRAD Students implement the following 2 functions.
// Undergrads can use the built-in barriers in pthreads.

// Call MyBarrier_Init once in main
void MyBarrier_Init()// you will likely need some parameters)
{
  P0 = nThreads + 1; // 16 threads + main thread
  count = nThreads + 1;
  pthread_mutex_init(&countMutex, 0); //Initialize Mutex used for counting threads
  // Create and initialize the localSense array, 1 entry per thread
  localSense = new bool[P0];
  for(int i = 0; i < P0; ++i)
  {
    localSense[i] = true;
  }

  globalSense = true; // initialize global sense.
}

// Each thread calls MyBarrier after completing the row-wise DFT
void MyBarrier(unsigned myId) // Again likely need parameters
{
    localSense[myId] = !localSense[myId]; // Toggle private thread variable
    if(threadCounter() == 1)
    {
      // All threads here, reset count and toggle globalSense
      count = P0;
      globalSense = localSense[myId];
    }
    else
    {
      while(globalSense != localSense[myId]) {  } //Spin
    }
}

int threadCounter()
{
  //cout<<"Inside threadCounter()"<<endl;
  pthread_mutex_lock(&countMutex);
	int myCount = count;
	count--;
	pthread_mutex_unlock(&countMutex);
	return myCount;

}

void reverseElements(Complex *h, int w1)
{
  //cout<<"Inside reverseElements"<<endl;
  Complex swapArray[w1];
  unsigned temp = 0;

  for (int row=0;row<1;row++)
  {
    for(int col=0;col<w1;col++)
    {
      temp = ReverseBits(col);
      swapArray[col] = *(h+(row*w1)+temp);
    }
    for(int x=0;x<w1;x++)
    *(h+x+(row*w1)) = swapArray[x];// ---> Refactor in previous for loop

  }
}

void transposeArray(Complex *h)
{
  Complex T;
	for(int row = 0;row < wd; row++)
	{
		for(int col = row; col < wd; col++)
		{
			int X=(row*wd)+col;
			int Y=(col*wd)+row;
			T=*(h+X);
			*(h+X)=*(h+Y);
			*(h+Y)=T;
		}
	}
}

void computeTwiddleFactors()
{
  for(int i = 0; i < wd/2; i++)
  {
    theta = (2*M_PI*i)/wd;
    // Forward twiddle factors
    W[i].real = cos(theta);
    W[i].imag = -1*sin(theta);

    // Inverse twiddle factors
    W_inv[i].real = cos(theta);
    W_inv[i].imag = sin(theta);
  }
}

void normalizeArray(Complex *h)
{
  for(int i = 0; i < imageSize; i++)
  {
    h[i].real = (double)h[i].real/wd;
    h[i].imag = (double)h[i].imag/wd;
  }
}

void adjustMag(Complex *h)
{
  for(int i = 0; i < imageSize; i++)
	{
		h[i] = h[i].Mag();
		if(h[i].real<0.5)
		h[i].real = 0;
	}
}

void Transform1D(Complex* h, int N)
{
  // Implement the efficient Danielson-Lanczos DFT here.
  // "h" is an input/output parameter
  // "N" is the size of the array (assume even power of 2)
  int width = wd;
  int split,delta;

  for(int i = 0; i < 1; i++)
  {
    for(int j = 2; j<=width; j = 2*j)
    {
      split = j/2;
      delta = N/j;

      for(int k = 0; k < width; k = k+j)
      {
        int v = (i*width)+k;
        int ind = 0;
        //cout<<"Am I going here?"<<endl;
        while(v<((i*width)+k+split))
        {
          Complex X,Y;
          X = *(h+v);
          Y = (*(h+v+split))*W[ind];    // Multiply with forward twiddle factor
        //  cout<<"Here"<<endl;
          *(h+v) = X+Y;
          *(h+v+split) = X-Y;
          ind=ind+delta;
          v++;
        }
      }
    }
  }
  //cout<<"Transform1D done for thread: "<<debugCount<<endl;
  //debugCount++;
}

void InverseTransform1D(Complex* h, int N)
{
  // Implement the efficient Danielson-Lanczos DFT here.
  // "h" is an input/output parameter
  // "N" is the size of the array (assume even power of 2)
  int width = wd;
  int split,delta;

  for(int i = 0; i < 1; i++)
  {
    for(int j = 2; j<=width; j = 2*j)
    {
      split = j/2;
      delta = N/j;

      for(int k = 0; k < width; k = k+j)
      {
        int v = (i*width)+k;
        int ind = 0;
        //cout<<"Am I going here?"<<endl;
        while(v<((i*width)+k+split))
        {
          Complex X,Y;
          X = *(h+v);
          Y = (*(h+v+split))*W_inv[ind]; // Mutiply with inverse twiddle factor
        //  cout<<"Here"<<endl;
          *(h+v) = X+Y;
          *(h+v+split) = X-Y;
          ind=ind+delta;
          v++;
        }
      }
    }
  }
  //cout<<"Transform1D done for thread: "<<debugCount<<endl;
  //debugCount++;
}

void* Transform2DTHread(void* v)
{ // This is the thread startign point.  "v" is the thread number
  // Calculate 1d DFT for assigned rows
  // wait for all to complete
  // Calculate 1d DFT for assigned columns
  // Decrement active count and signal main if all complete
  unsigned long threadId = (unsigned long)v;
  int rowsPerThread = imageHeight/nThreads;
  //cout<<"rows: "<<rowsPerThread<<endl;
  for(int i = 0; i < rowsPerThread; i++)
  {
    reverseElements(&d[rowsPerThread*wd*threadId + i*wd],wd); // --> Check
    Transform1D(&d[rowsPerThread*wd*threadId + i*wd],wd);
  }
  MyBarrier(threadId);
  return 0;
}

void* InverseTransform2DTHread(void* v)
{ // This is the thread startign point.  "v" is the thread number
  // Calculate 1d DFT for assigned rows
  // wait for all to complete
  // Calculate 1d DFT for assigned columns
  // Decrement active count and signal main if all complete
  unsigned long threadId = (unsigned long)v;
  int rowsPerThread = imageHeight/nThreads;
  //cout<<"rows: "<<rowsPerThread<<endl;
  for(int i = 0; i < rowsPerThread; i++)
  {
    reverseElements(&d[rowsPerThread*wd*threadId + i*wd],wd); // --> Complete.
    InverseTransform1D(&d[rowsPerThread*wd*threadId + i*wd],wd);
  }
  MyBarrier(threadId);
  return 0;
}

void* InverseTransform2DTHreadCond(void* v)
{ // This is the thread startign point.  "v" is the thread number
  // Calculate 1d DFT for assigned rows
  // wait for all to complete
  // Calculate 1d DFT for assigned columns
  // Decrement active count and signal main if all complete
  unsigned long threadId = (unsigned long)v;
  int rowsPerThread = imageHeight/nThreads;
  //cout<<"rows: "<<rowsPerThread<<endl;
  for(int i = 0; i < rowsPerThread; i++)
  {
    reverseElements(&d[rowsPerThread*wd*threadId + i*wd],wd); // --> Complete.
    InverseTransform1D(&d[rowsPerThread*wd*threadId + i*wd],wd);
  }

  // Wait for all threads to complete.
  pthread_mutex_lock(&startCountMutex);
  threadCount--;
  if(threadCount == 0)
  {
    //Notify main if last to exit
    pthread_mutex_unlock(&startCountMutex);
    pthread_mutex_lock(&exitMutex);
    pthread_cond_signal(&exitCond);
    pthread_mutex_unlock(&exitMutex);
  }
  else
  {
    pthread_mutex_unlock(&startCountMutex);
  }
  return 0;
}


int main(int argc, char** argv)
{

  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  InputImage image(fn.c_str());  // Create the helper object for reading the image

  ht = image.GetHeight();
  imageHeight = ht;
  wd = image.GetWidth();
  d = image.GetImageData();
  imageSize = ht*wd;

  computeTwiddleFactors(); // Computing twiddle factors for forward and inverse DFT

  MyBarrier_Init();

  pthread_mutex_init(&exitMutex,0);
  pthread_cond_init(&exitCond,0);
  pthread_mutex_init(&startCountMutex,0);

  //cout<<"Apple"<<endl;
  // ======================= START FORWARD DFT ========================= //
  cout<<"Starting Forward DFT.."<<endl;

  pthread_t threads[nThreads];
  int th;
  // Step 1: 16 threads perform row transformation. Each thread works on 64 rows
  for(th = 0; th < nThreads;th++)
  {
    pthread_create(&threads[th], 0, Transform2DTHread, (void *)th);
    //cout<<"thread No.: "<<th<<endl;
  }
  MyBarrier(16);

  cout<<"1D transformation complete .."<<endl;
  image.SaveImageData("MyAfter1d.txt",d,wd,ht);

  // Step 2: Transpose the output after row transformation
  transposeArray(d);

  // Step 3: Now, Each thread works on 64 columns each.
  for(th = 0; th < nThreads;th++)
  {
    pthread_create(&threads[th], 0, Transform2DTHread, (void *)th);
    //cout<<"thread: "<<th<<endl;
  }
  MyBarrier(16);

  // Step 4: Transpose the output of column transformation to obtain 2D-DFT.
  transposeArray(d);

  cout<<"2D transformation complete.."<<endl;
  image.SaveImageData("Tower-DFT2D.txt",d,wd,ht);


  // ======================= START INVERSE DFT ========================= //
  cout<<"Starting inverse DFT.."<<endl;

  // Step 5: 16 threads perform row transformation on 64 rows each.
  for(th = 0; th < nThreads;th++)
  {
    pthread_create(&threads[th], 0, InverseTransform2DTHread, (void *)th);
    //cout<<"thread: "<<th<<endl;
  }
  MyBarrier(16);

  // Step 6: Transpose and Normalize the output of inverse row transform.
  transposeArray(d);
  normalizeArray(d);  // Normalize after inverse.

  // Transform using pthread mutex condition.
  pthread_mutex_lock(&exitMutex);
  threadCount = nThreads;
  // Step 7: Perform column transform after normalizing and transposing.
  for(th = 0; th < nThreads;th++)
  {
    pthread_create(&threads[th], 0, InverseTransform2DTHreadCond, (void *)th);
    //cout<<"thread: "<<th<<endl;
  }
  pthread_cond_wait(&exitCond, &exitMutex);  // Wait till all threads are done.

  // Step 8: Transpose and normalize the resulting output to get the inverse 2D-DFT.
  transposeArray(d);
  normalizeArray(d);  // Normalize aftr inverse
  adjustMag(d);

  image.SaveImageData("TowerInverse.txt",d,wd,ht);
  cout<<"Inverse 2D transform complete .."<<endl;
  //cout<<"Pineapple"<<endl;
  return 0;
}
