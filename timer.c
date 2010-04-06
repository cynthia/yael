/* Available only for Linux at this moment */
#if ( (defined(unix) || defined(__unix))  )

#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

#ifdef RUSAGE_SELF

#include <stdlib.h>
#include "timer.h"


it_timer_t * timer_new()
{
  it_timer_t * timer = (it_timer_t *) malloc( sizeof(it_timer_t) );
  timer_rtz( timer );
  return timer;
}


void timer_rtz( it_timer_t * timer )
{
  timer->status        = 0;
  timer->amount_wall   = 0.0;
  timer->amount_user   = 0.0;
  timer->amount_system = 0.0;
}


void timer_free( it_timer_t * timer )
{
  free( timer );
}


void timer_on( it_timer_t * timer ) 
{
  timer->status = 1;

  /* store the current time for later use */
  gettimeofday( &timer->time_last_wall, NULL );
  getrusage( RUSAGE_SELF, &timer->time_last_cpu );
}


void timer_off( it_timer_t * timer ) 
{
  timer->status = 0;

  /* store the current time for later use */
  gettimeofday (&timer->time_now_wall, NULL) ;
  getrusage(RUSAGE_SELF, &timer->time_now_cpu);
  
  /* update the current amounts */
  timer->amount_wall += ((double) (timer->time_now_wall.tv_sec 
				   - timer->time_last_wall.tv_sec) 
			 + (double) (timer->time_now_wall.tv_usec 
				     - timer->time_last_wall.tv_usec) / 1000000);
  timer->amount_user += ((double) (timer->time_now_cpu.ru_utime.tv_sec 
				   - timer->time_last_cpu.ru_utime.tv_sec) 
			 + (double) (timer->time_now_cpu.ru_utime.tv_usec 
				     - timer->time_last_cpu.ru_utime.tv_usec) / 1000000);
  timer->amount_system += ((double) (timer->time_now_cpu.ru_stime.tv_sec 
				     - timer->time_last_cpu.ru_stime.tv_sec) 
			   + (double) (timer->time_now_cpu.ru_stime.tv_usec 
				       - timer->time_last_cpu.ru_stime.tv_usec) / 1000000);
}


/* Return the different kind of usages */
double timer_wall( it_timer_t * timer )
{
  return timer->amount_wall;
}


double timer_user( it_timer_t * timer )
{
  return timer->amount_user;
}


double timer_system( it_timer_t * timer )
{
  return timer->amount_system;
}


double timer_cpu( it_timer_t * timer )
{
  return timer->amount_user+timer->amount_system;
}

#endif
#endif
