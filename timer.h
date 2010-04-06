/*
  Timer, for linux only
*/

#ifndef __timer_h_
#define __timer_h_

#include <sys/time.h>
#include <sys/resource.h>

typedef struct {
  unsigned short    status;           /* timing status : 0 <=> idle, 1 <=> active */
  
  struct timeval    time_now_wall;    /* now registered wall time */
  struct timeval    time_last_wall;   /* last registered wall time */
  double            amount_wall;      /* amount of wall time */
  
  struct rusage     time_now_cpu;     /* now registered cpu (user + syst) time  */
  struct rusage     time_last_cpu;    /* last registered cpu (user + syst) time */
  double            amount_user;      /* amount of user time */
  double            amount_system;    /* amount of system time */
} it_timer_t;

/* Initialization of the timer */
it_timer_t * timer_new();

/* Reinitialization of the timer */
void timer_rtz( it_timer_t * timer );
  
/* Free the timer */
void timer_free( it_timer_t * timer );
  
/* Set the timer on or off */
void timer_on( it_timer_t * timer );
void timer_off( it_timer_t * timer );

/* Return the different kind of usages (CPU is system+user)*/
double timer_wall( it_timer_t * timer );
double timer_user( it_timer_t * timer );
double timer_system( it_timer_t * timer );
double timer_cpu( it_timer_t * timer );

#endif
