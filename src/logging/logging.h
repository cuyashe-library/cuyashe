#ifndef LOGGING_H
#define LOGGING_H

#include <iostream>
#include <string.h>
#include "log.h"

// Initiates a new log
void log_init();
void log_init(std::string s);
void log_notice(std::string s);
void log_debug(std::string s);
void log_warn(std::string s);
void log_error(std::string s);


#endif