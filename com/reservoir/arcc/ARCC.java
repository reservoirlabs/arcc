// DOE-GPU rights

// $Id$

// Copyright (C) 2010 Reservoir Labs, Inc. All rights reserved. 

package com.reservoir.arcc;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import java.util.Collection;

/** 
 * Java toolbox for the Autotuned R-Stream C Compiler (ARCC). The
 * main constants in the interface between ARCC and R-Stream are
 * defined here.
 * 
 * @author Benoit Meister <meister@reservoir.com>
 * @version $Revision$ $Date$
 */

public class ARCC {

    // Name of the environment variable that contains the file name to
    // write meta-data to.
    private static final String ARCC_METADATA_ENV = "ARCC_METADATA";

    // Name of the environment variable that contains the ARCC mode
    // (consume, produce or none)
    private static final String ARCC_MODE_ENV = "ARCC_MODE";

    // Name of the environment variable that contains the ARCC
    // performance counting type (rough or precise)
    private static final String ARCC_PERF_ENV = "ARCC_PERF";

    // Name of the environment variable that contains the file name to
    // write the measured performance number to.
    private static final String ARCC_PERF_FILE_ENV = "ARCC_PERFFILE";

    // Name of the environment variable that indicates the use mode of
    // the tactic options is to be tuned (all, default, or
    // user-specified)
    private static final String ARCC_OPTION_USE_MODE_ENV = "ARCC_OPTIONUSEMODE";

    // Name of the environment variable that contains the file name to
    // write the list of autotunable tactic options.
    private static final String ARCC_OPTION_FILE_ENV = "ARCC_OPTIONFILE";

    /** 
     * No mode has been defined by ARCC, which is interpeted as "there is no
     * ARCC running"
     */
    public static final int NO_ARCC = 0;

    /** ARCC is in "produce meta-data" mode */
    public static final int PRODUCE = 1;

    /** ARCC is in "consume options" mode */
    public static final int CONSUME = 2;

    /** Test if ARCC is present, from an ARCC mode */
    public static final boolean isActive() {
	return getMode() != NO_ARCC;
    }

    /** Test if ARCC is in "produce" mode */
    public static final boolean isProduce() {
	return getMode() == PRODUCE;
    }

    /** Test if ARCC is in "consume" mode */
    public static final boolean isConsume() {
	return getMode() == CONSUME;
    }

    /** 
     * Return the current auto-tuning mode, which is either "PRODUCE"
     * or "CONSUME". If ARCC hasn't defined a mode, then returns an
     * empty string.
     */
    private static int getMode() {
	String mode = System.getenv(ARCC_MODE_ENV);
	if (mode == null || mode.isEmpty()) {
	    return NO_ARCC;
	} else if (mode.compareToIgnoreCase("produce") == 0) {
	    String metadata = System.getenv(ARCC_METADATA_ENV);
	    assert (metadata != null && !metadata.isEmpty()) : 
	    "Missing/empty meta-data input file defined in $" + ARCC_METADATA_ENV;
	    return PRODUCE;
	} else if (mode.compareToIgnoreCase("consume") == 0) {
	    return CONSUME;
	} else {
	    throw new Error("$" + ARCC_MODE_ENV + " mis-defined: " + mode +
			    ".\n Must be \"produce\", \"consume\" or nothing");
	}
    }

    /** Tune "all" autotunable tactic options */
    public static final int ALL_OPTIONS = 0;

    /** Tune "default" autotunable tactic options */
    public static final int DEFAULT_OPTIONS = 1;

    /** Tune only the "user-specified" autotunable tactic options */
    public static final int USER_SPECIFIED_OPTIONS = 2;

    /** Test if "all" autotunable tactic options are to be tuned */
    public static final boolean useAllOptions() {
	return optionUseMode() == ALL_OPTIONS;
    }

    /** Test if the "default" autotunable tactic options are to be tuned */
    public static final boolean useDefaultOptions() {
	return optionUseMode() == DEFAULT_OPTIONS;
    }

    /** Test if only "user-specified" autotunable tactic options are to be tuned */
    public static final boolean useUserSpecifiedOptions() {
	return optionUseMode() == USER_SPECIFIED_OPTIONS;
    }

    /**
     * Return the use mode of the autotunable tactic options that is
     * to be tuned. The use mode is either "all", "default" or
     * "user-specified".
     */
    private static int optionUseMode() {
	String opts = System.getenv(ARCC_OPTION_USE_MODE_ENV);
	assert (opts != null && !opts.isEmpty()) : "$" + ARCC_OPTION_USE_MODE_ENV + 
	    " ill-defined: " + opts + 
	    ".\n Must be \"all\", \"default\", or \"user-specified\".";
	if (opts.compareToIgnoreCase("all") == 0) {
	    return ALL_OPTIONS;
	} else if (opts.compareToIgnoreCase("default") == 0) {
	    return DEFAULT_OPTIONS;
	} else if (opts.compareToIgnoreCase("user-specified") == 0) {
	    return USER_SPECIFIED_OPTIONS;
	} else {
	    throw new Error("$" + ARCC_OPTION_USE_MODE_ENV + " mis-defined: " + opts +
			    ".\n Must be \"all\", \"default\", or \"user-specified\".");
	}
    }
    
    /**
     * Save the given options string into a file from which later ARCC
     * will read the autotunable tactic options.
     */
    public static void saveAutotunableOptions(String opt_str) throws IOException {
	// Get the filename for saving the list of autotunable tactic options	
	String opt_fname = System.getenv(ARCC_OPTION_FILE_ENV);
	// Make sure it's given by ARCC
	assert (opt_fname != null && !opt_fname.isEmpty()) : "Variable $" + 
	    ARCC_OPTION_FILE_ENV + " undefined";
	// Make sure the path is absolute
	assert opt_fname.indexOf(File.separator) != -1 : "Absolute path required for $" + 
	    ARCC_OPTION_FILE_ENV + " (" + opt_fname + ")," + File.separator + 
	    " not found";
	// Create and write the file
	File out_file = new File(opt_fname);
	out_file.createNewFile();
	FileWriter f = new FileWriter(out_file, false);
	f.write(opt_str);
	f.close();
    }

    /** Open the ARCC meta-data file and append {@code metadata} to it. */
    public static void writeMetaData(String metadata) throws IOException {
	// Get the meta-data filename from ARCC
	String md_fname = System.getenv(ARCC_METADATA_ENV);
	assert (md_fname != null && !md_fname.isEmpty()): "Variable $" 
	    + ARCC_METADATA_ENV + " undefined";
	assert md_fname.indexOf(File.separator) != -1:
	"Absolute path required for $" + ARCC_METADATA_ENV + " (" + md_fname + ")," + 
	    File.separator + " not found";
	// Create and write the file (in append mode)
	File out_file = new File(md_fname);
	out_file.createNewFile();
	FileWriter f = new FileWriter(out_file, true);
	f.append(metadata);
	f.close();
    }

    /** Test if the precise time measurement (for ARCC) is used. */
    public static boolean preciseTiming() {
	String perf_type = System.getenv(ARCC_PERF_ENV);
	assert (perf_type != null && !perf_type.isEmpty()) : "Variable $" + 
	    ARCC_PERF_ENV + " undefined";
	if (perf_type.compareToIgnoreCase("rough") == 0) {
	    return false;
	} else if (perf_type.compareToIgnoreCase("precise") == 0) {
	    return true;
	} else 
	    throw new Error("$" + ARCC_PERF_ENV + " ill-defined: " + perf_type + 
			    ".\n Must be \"rough\" or \"precise\".");
    }

    /** 
     * Return the file name to write the measured performance number to.
     */
    public static String timingOutputFilename() {
	String perf_fname = System.getenv(ARCC_PERF_FILE_ENV);
	assert (perf_fname != null && !perf_fname.isEmpty()): "Variable $" + 
	    ARCC_PERF_FILE_ENV + " undefined";
	assert perf_fname.indexOf(File.separator)!=-1: 
	"Absolute path required for $" + ARCC_PERF_FILE_ENV + " (" + perf_fname + ")," + 
	    File.separator+" not found";
	return perf_fname;
    }

    // _________________________ Meta-data syntax elements _________________________
    //
    //

    /** String for starting an option syntax line */
    public static final String OPTION_BEGIN = " option = \"";

    /** String for ending an option syntax line */
    public static final String OPTION_END = "\";\n";

    /** Starts a constraint declaration */
    public static final String CONSTRAINT_BEGIN = " constraint = ";

    /** Ends a constraint declaration */
    public static final String CONSTRAINT_END = ";\n";

    /** Ends the declaration of a tactic aut-tuning entry */
    public static final String END = "}\n";

    /** 
     * Starts the declaration of tactic auto-tuning entry that has ID
     * {@code ID}
     */
    public static final String BEGIN(String ID) {
	return ID + " {\n";
    }

    /** 
     * Produces the string that defines the possible values of an ARCC
     * search variable.
     * @param vname name of the search variable
     * @param values set of possible values for the variable
     */
    public static final String varDef(String vname, Collection<? extends Object> values) {
	StringBuilder sb = new StringBuilder(" var " + vname + " = [");
	boolean first_v = true;
	for (Object val: values) {
	    if (first_v) {
		first_v = false;
	    } else {
		sb.append(",");
	    }
	    if (val instanceof Number) {
		sb.append(val);
	    } else {
		sb.append("\"" + val + "\"");
	    }
	}
	sb.append("]; \n");
	return sb.toString();
    }

    /** 
     * Produces the string that defines the possible values of an ARCC
     * search variable.
     * @param vname name of the search variable
     * @param values set of possible values for the variable
     */
    public static final String varDef(String vname, String[] values) {
	StringBuilder sb = new StringBuilder(" var " + vname + " = [");
	boolean first_v = true;
	for (String val: values) {
	    if (first_v) {
		first_v = false;
	    } else {
		sb.append(",");
	    }
	    sb.append(val);
	}
	sb.append("]; \n");
	return sb.toString();
    }

    // ______________________________ Various utils ______________________________
    //
    //

    /** 
     * Return a string representation of the given array of long
     * numbers.
     */
    public static String longArray(long[] vals) {
	if (vals.length == 0) 
	    return "";
	StringBuilder sb = new StringBuilder();
        sb.append("{");
	boolean is_first = true;
        for (long v : vals) {
	    if (is_first) {
		is_first = false;
	    } else {
		sb.append("-");
	    }
            sb.append(v);
        }
        sb.append("}");
        return sb.toString();
    }

     /** 
     * Return a string representation of the given array of long
     * numbers.
     */
    public static String intArray(int [] vals) {
	if (vals.length == 0) 
	    return "";
	StringBuilder sb = new StringBuilder();
        sb.append("{");
	boolean is_first = true;
        for (long v : vals) {
	    if (is_first) {
		is_first = false;
	    } else {
		sb.append("-");
	    }
            sb.append(v);
        }
        sb.append("}");
        return sb.toString();
    }
}

// Local Variables:
// mode: java
// c-basic-offset: 4
// End:
