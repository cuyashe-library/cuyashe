/**
 * cuYASHE
 * Copyright (C) 2015-2016 cuYASHE Authors
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "logging.h"
// #define DEBUG


#ifdef DEBUG
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

void log_init(){

}

void log_init(std::string s){
	
}

void log_notice(std::string s){
}
void log_debug(std::string s){
}
void log_warn(std::string s){
}
void log_error(std::string s){
}
#endif