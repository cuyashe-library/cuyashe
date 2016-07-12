#include "logging.h"
#define DEBUG

void log_init(){
	// log_init(LL_DEBUG);
	log_notice("");
	log_debug("");
	log_warn("");
	log_error("");
	log_notice("...Starting...");
	log_debug("...Starting...");
	log_warn("...Starting...");
	log_error("...Starting...");
}

void log_init(std::string s){
	log_init(LL_DEBUG, s.c_str(), "./log/");
	log_notice("");
	log_debug("");
	log_warn("");
	log_error("");
	log_notice("...Starting...");
	log_debug("...Starting...");
	log_warn("...Starting...");
	log_error("...Starting...");
}

#ifdef DEBUG
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
#else

void log_notice(std::string s){
}
void log_debug(std::string s){
}
void log_warn(std::string s){
}
void log_error(std::string s){
}
#endif