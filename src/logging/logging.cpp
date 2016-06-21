#include "logging.h"

void log_init(){
	log_init(LL_TRACE, "default.log", "./log/");
}

void log_init(std::string s){
	log_init(LL_TRACE, s.c_str(), "./log/");
}

void log_notice(std::string s){
	LOG_NOTICE("%s",s.c_str());
}
void log_debug(std::string s){
	LOG_DEBUG("%s",s.c_str());
}
void log_warn(std::string s){
	LOG_WARN("%s",s.c_str());
}
void log_error(std::string s){
	LOG_ERROR("%s",s.c_str());
}